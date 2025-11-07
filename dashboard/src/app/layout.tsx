// src/app/layout.tsx (Updated)
import type { Metadata } from "next"; 
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
 

export const metadata: Metadata = {
  title: "Missing Person Dashboard",
  description: "Admin dashboard for missing person management",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body  >
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
