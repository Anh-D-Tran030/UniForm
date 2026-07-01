import type { Metadata } from "next";
import { Bricolage_Grotesque, Hanken_Grotesk, JetBrains_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";

const displayFont = Bricolage_Grotesque({
  variable: "--font-display",
  subsets: ["latin"],
  display: "swap",
  weight: ["400", "500", "600", "700", "800"],
});

const sansFont = Hanken_Grotesk({
  variable: "--font-sans-ui",
  subsets: ["latin"],
  display: "swap",
});

const monoFont = JetBrains_Mono({
  variable: "--font-mono-data",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "UniForm",
  description: "Document matching and extraction workspace",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${displayFont.variable} ${sansFont.variable} ${monoFont.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full bg-[var(--surface-subtle)] font-sans text-[var(--ink)]">
        <Script
          id="uniform-theme-init"
          strategy="beforeInteractive"
          dangerouslySetInnerHTML={{
            __html:
              "try{var t=localStorage.getItem('uniform-theme')==='light'?'light':'dark';document.documentElement.dataset.theme=t;}catch(e){}",
          }}
        />
        {children}
      </body>
    </html>
  );
}
