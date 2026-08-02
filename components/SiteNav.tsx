"use client";

import { usePathname } from "next/navigation";

/**
 * Shared navigation.
 *
 * Without it the three routes were unreachable from one another: the deployed site looked
 * like a single search box, and /journey and /setup existed but nothing linked to them.
 * Pages that cannot be navigated to are, from a user's point of view, not deployed.
 */
const LINKS = [
  { href: "/", label: "Overview", hint: "how it works, what it scores, and why" },
  { href: "/search", label: "Search", hint: "find passages and compare papers" },
];

export default function SiteNav() {
  const pathname = usePathname();
  return (
    <nav
      aria-label="Primary"
      className="sticky top-0 z-50 border-b border-white/8 bg-[#05070c]/80 backdrop-blur-md"
    >
      <div className="mx-auto flex max-w-5xl items-center gap-1 px-6 py-3">
        <a href="/" className="mr-4 text-sm font-medium tracking-tight text-white">
          Onco<span className="text-cyan-300">Lens</span>
        </a>
        {LINKS.map((l) => {
          const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
          return (
            <a
              key={l.href}
              href={l.href}
              title={l.hint}
              aria-current={active ? "page" : undefined}
              className={`rounded-md px-3 py-1.5 text-[13px] transition-colors ${
                active
                  ? "bg-white/10 text-white"
                  : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
              }`}
            >
              {l.label}
            </a>
          );
        })}
        <a
          href="https://github.com/jcl347/oncolens"
          target="_blank"
          rel="noreferrer"
          className="ml-auto text-[13px] text-slate-500 transition-colors hover:text-slate-300"
        >
          source ↗
        </a>
      </div>
    </nav>
  );
}
