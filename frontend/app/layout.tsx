import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";
import localFont from "next/font/local";

const shabnam = localFont({
  src: [
    { path: "../fonts/shabnam/Shabnam-Thin.woff2", weight: "100", style: "normal" },
    { path: "../fonts/shabnam/Shabnam-Light.woff2", weight: "300", style: "normal" },
    { path: "../fonts/shabnam/Shabnam.woff2", weight: "400", style: "normal" },
    { path: "../fonts/shabnam/Shabnam-Medium.woff2", weight: "500", style: "normal" },
    { path: "../fonts/shabnam/Shabnam-Bold.woff2", weight: "700", style: "normal" },
  ],
  variable: "--font-shabnam",
  display: "swap",
});

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "TaxYar",
  description: "Legal assistant for Iranian Tax Laws",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fa" className="dark">
      <body
        className={`${shabnam.variable} antialiased font-sans`}
      >
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}
