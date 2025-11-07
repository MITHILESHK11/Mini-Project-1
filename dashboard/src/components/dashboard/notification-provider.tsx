// src/components/dashboard/notification-provider.tsx
"use client";

import { useEffect } from "react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import { CheckCircle } from "lucide-react";

export function NotificationProvider() {
  const { notification, clearNotification } = useWebSocket();
  const router = useRouter();

  useEffect(() => {
    if (notification?.event === "child_found" && notification.child) {
      const { child } = notification;
      
      toast.success("Child Found!", {
        description: `${child.name} (ID: ${child.id}) has been found!`,
        icon: <CheckCircle className="w-5 h-5" />,
        duration: 10000,
        action: {
          label: "View",
          onClick: () => router.push("/recently-found"),
        },
      });

      // Refresh the page data
      router.refresh();
      clearNotification();
    }
  }, [notification, clearNotification, router]);

  return null;
}
