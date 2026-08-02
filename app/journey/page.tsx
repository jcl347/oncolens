import fs from "node:fs";
import path from "node:path";
import JourneyClient from "./JourneyClient";

export const metadata = {
  title: "OncoLens — how it was built and how it is measured",
  description:
    "The technical journey: what was tried, what failed, and the measurement that "
    + "decided each design choice.",
};

// Read at build time from the artifact scripts/build_journey_data.py produces. If it is
// missing the page says so, rather than falling back to plausible-looking constants —
// which is the failure mode the whole project is built to avoid.
function loadJourney() {
  try {
    const p = path.join(process.cwd(), "public", "journey.json");
    return JSON.parse(fs.readFileSync(p, "utf-8"));
  } catch {
    return null;
  }
}

export default function JourneyPage() {
  const data = loadJourney();

  if (!data) {
    return (
      <main className="mx-auto max-w-2xl px-6 py-32 text-slate-300">
        <h1 className="text-2xl font-medium text-white">Measurements unavailable</h1>
        <p className="mt-4 text-sm leading-relaxed text-slate-400">
          <code className="text-cyan-300">public/journey.json</code> has not been generated.
          Run <code className="text-cyan-300">python scripts/build_journey_data.py</code> to
          produce it. This page deliberately shows nothing rather than placeholder numbers:
          a metric you cannot trace to a command is worse than no metric.
        </p>
      </main>
    );
  }

  return <JourneyClient data={data} />;
}
