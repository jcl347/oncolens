#!/usr/bin/env node
/**
 * Node bridge to @vercel/blob, called by the Python ingestion.
 *
 * Why this exists: Vercel Blob's REST API rejects every upload to a **private** store with
 * "Cannot use public access on a private store", and no combination of x-access /
 * x-blob-access / api-version headers gets past it — private stores evidently use a
 * protocol the public REST surface does not expose. The official SDK handles it, so rather
 * than reverse-engineering an undocumented handshake (which would break silently the next
 * time Vercel changes it), Python shells out to the SDK.
 *
 * Protocol: one JSON request per line on stdin, one JSON response per line on stdout.
 * Batching over a single process matters — spawning Node per article would dominate
 * runtime on a multi-thousand-document ingest.
 *
 *   {"op":"put","pathname":"pmc/txt/PMC123.1.txt","content":"…","access":"private"}
 *   {"op":"del","urls":["https://…"]}
 *   {"op":"get","url":"https://…"}
 */

import { put, del, head } from "@vercel/blob";
import { createInterface } from "node:readline";

const token = process.env.BLOB_READ_WRITE_TOKEN;
if (!token) {
  process.stdout.write(
    JSON.stringify({ ok: false, error: "BLOB_READ_WRITE_TOKEN not set" }) + "\n"
  );
  process.exit(1);
}

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });

for await (const line of rl) {
  const trimmed = line.trim();
  if (!trimmed) continue;

  let req;
  try {
    req = JSON.parse(trimmed);
  } catch (e) {
    process.stdout.write(JSON.stringify({ ok: false, error: `bad JSON: ${e.message}` }) + "\n");
    continue;
  }

  try {
    if (req.op === "put") {
      const blob = await put(req.pathname, req.content, {
        // Default to private: the store is private, and requesting public 400s. Callers
        // may override explicitly.
        access: req.access ?? "private",
        token,
        contentType: req.contentType ?? "text/plain; charset=utf-8",
        // Deterministic paths so re-ingesting overwrites instead of accumulating copies —
        // which matters when a long job is resumed after a partial failure.
        addRandomSuffix: false,
        allowOverwrite: true,
      });
      process.stdout.write(
        JSON.stringify({
          ok: true,
          url: blob.url,
          downloadUrl: blob.downloadUrl ?? null,
          pathname: blob.pathname,
          size: Buffer.byteLength(req.content, "utf8"),
        }) + "\n"
      );
    } else if (req.op === "del") {
      await del(req.urls, { token });
      process.stdout.write(JSON.stringify({ ok: true, deleted: req.urls.length }) + "\n");
    } else if (req.op === "head") {
      const meta = await head(req.url, { token });
      process.stdout.write(JSON.stringify({ ok: true, size: meta.size, pathname: meta.pathname }) + "\n");
    } else {
      process.stdout.write(JSON.stringify({ ok: false, error: `unknown op ${req.op}` }) + "\n");
    }
  } catch (e) {
    // Never let one failed object kill the batch; the caller decides whether to continue.
    process.stdout.write(JSON.stringify({ ok: false, error: String(e.message ?? e).slice(0, 300) }) + "\n");
  }
}
