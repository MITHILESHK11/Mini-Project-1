// src/app/(dashboard)/layout.tsx (Updated)
import { Sidebar } from "@/components/dashboard/sidebar";
import { NotificationProvider } from "@/components/dashboard/notification-provider";
import { Toaster } from "@/components/ui/sonner";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="h-screen flex">
      <div className="hidden md:flex w-64 flex-col fixed inset-y-0">
        <Sidebar />
      </div>
      <div className="flex-1 md:pl-64">
        {children}
      </div>
      <NotificationProvider />
      <Toaster richColors position="top-right" />
    </div>
  );
}
