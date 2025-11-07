// src/components/NotificationBanner.tsx
"use client";

import { useEffect } from "react";
import { WebSocketNotification } from "@/lib/types";

interface NotificationBannerProps {
  notification: WebSocketNotification | null;
  onClose: () => void;
  onRefresh: () => void;
}

export default function NotificationBanner({
  notification,
  onClose,
  onRefresh,
}: NotificationBannerProps) {
  useEffect(() => {
    if (notification?.event === "child_found") {
      // Auto-refresh data when child is found
      onRefresh();
      
      // Auto-dismiss after 10 seconds
      const timer = setTimeout(onClose, 10000);
      return () => clearTimeout(timer);
    }
  }, [notification, onClose, onRefresh]);

  if (!notification) return null;

  if (notification.event === "child_found" && notification.child) {
    return (
      <div className="fixed top-4 right-4 z-50 max-w-md animate-slide-in">
        <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded-lg shadow-lg">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <span className="text-2xl">🟢</span>
            </div>
            <div className="ml-3 flex-1">
              <h3 className="text-sm font-semibold text-green-800">
                Child Found!
              </h3>
              <p className="text-sm text-green-700 mt-1">
                <strong>{notification.child.name}</strong> (ID: {notification.child.id})
              </p>
            </div>
            <button
              onClick={onClose}
              className="ml-3 text-green-500 hover:text-green-700"
            >
              ✕
            </button>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
