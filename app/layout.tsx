import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OncoLens — passage-grounded oncology literature retrieval",
  description:
    "Search oncology papers and grants, see the exact passage where a concept was mentioned, and compare papers on the same technical dimensions.",
};

import SiteNav from "../components/SiteNav";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">
        <SiteNav />
        {children}
      </body>
    </html>
  );
}
