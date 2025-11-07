// src/app/(dashboard)/notifications/page.tsx
"use client";

import { Header } from "@/components/dashboard/header";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Bell, CheckCircle, Clock, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useWebSocket } from "@/hooks/useWebSocket";

interface Notification {
  id: string;
  title: string;
  description: string;
  time: string;
  timestamp: number;
  type: "found" | "system" | "alert";
  read: boolean;
  childData?: {
    id: number;
    name: string;
    age: number;
    contact: string;
  };
}

export default function NotificationsPage() {
  const { notification: wsNotification } = useWebSocket();
  const [notifications, setNotifications] = useState<Notification[]>(() => {
    // Load from localStorage on mount
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("notifications");
      return saved ? JSON.parse(saved) : [];
    }
    return [];
  });

  const unreadCount = notifications.filter((n) => !n.read).length;

  // Listen for WebSocket notifications
  useEffect(() => {
    if (wsNotification?.event === "child_found" && wsNotification.child) {
      const { child } = wsNotification;
      
      const newNotification: Notification = {
        id: `ws-${Date.now()}`,
        title: "Child Found!",
        description: `${child.name} (ID: ${child.id}) has been successfully recovered`,
        time: "Just now",
        timestamp: Date.now(),
        type: "found",
        read: false,
        childData: {
          id: child.id,
          name: child.name,
          age: child.age,
          contact: child.contact,
        },
      };

      setNotifications((prev) => {
        const updated = [newNotification, ...prev];
        // Save to localStorage
        localStorage.setItem("notifications", JSON.stringify(updated));
        return updated;
      });
    }
  }, [wsNotification]);

  // Update relative time every minute
  useEffect(() => {
    const interval = setInterval(() => {
      setNotifications((prev) =>
        prev.map((n) => ({
          ...n,
          time: getRelativeTime(n.timestamp),
        }))
      );
    }, 60000); // Update every minute

    return () => clearInterval(interval);
  }, []);

  const markAsRead = (id: string) => {
    setNotifications((prev) => {
      const updated = prev.map((n) => (n.id === id ? { ...n, read: true } : n));
      localStorage.setItem("notifications", JSON.stringify(updated));
      return updated;
    });
  };

  const markAllAsRead = () => {
    setNotifications((prev) => {
      const updated = prev.map((n) => ({ ...n, read: true }));
      localStorage.setItem("notifications", JSON.stringify(updated));
      return updated;
    });
  };

  const clearAll = () => {
    if (confirm("Are you sure you want to clear all notifications?")) {
      setNotifications([]);
      localStorage.removeItem("notifications");
    }
  };

  const deleteNotification = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setNotifications((prev) => {
      const updated = prev.filter((n) => n.id !== id);
      localStorage.setItem("notifications", JSON.stringify(updated));
      return updated;
    });
  };

  return (
    <div className="flex flex-col h-screen">
      <Header
        title="Notifications"
        description={`${unreadCount} unread notification${unreadCount !== 1 ? "s" : ""}`}
      />
      
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-3xl">
          {/* Action Buttons */}
          {notifications.length > 0 && (
            <div className="flex gap-2 mb-4">
              {unreadCount > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={markAllAsRead}
                >
                  Mark all as read
                </Button>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={clearAll}
                className="ml-auto"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear all
              </Button>
            </div>
          )}

          {/* Notifications List */}
          <div className="space-y-3">
            {notifications.length > 0 ? (
              notifications.map((notification) => (
                <Card
                  key={notification.id}
                  className={`cursor-pointer transition-colors hover:bg-accent/20 ${
                    !notification.read ? "bg-accent/50 border-primary/50" : ""
                  }`}
                  onClick={() => markAsRead(notification.id)}
                >
                  <CardContent className="p-4 flex items-start gap-4">
                    <div
                      className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
                        notification.type === "found"
                          ? "bg-green-500/10"
                          : "bg-blue-500/10"
                      }`}
                    >
                      {notification.type === "found" ? (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      ) : (
                        <Bell className="w-5 h-5 text-blue-500" />
                      )}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2">
                        <h4 className="font-semibold">{notification.title}</h4>
                        <div className="flex items-center gap-2">
                          {!notification.read && (
                            <Badge variant="default" className="whitespace-nowrap">
                              New
                            </Badge>
                          )}
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={(e) => deleteNotification(notification.id, e)}
                          >
                            <Trash2 className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        {notification.description}
                      </p>
                      
                      {/* Child Data if available */}
                      {notification.childData && (
                        <div className="mt-2 p-2 bg-muted/50 rounded text-xs space-y-1">
                          <p><strong>Age:</strong> {notification.childData.age}</p>
                          <p><strong>Contact:</strong> {notification.childData.contact}</p>
                        </div>
                      )}
                      
                      <div className="flex items-center gap-1 text-xs text-muted-foreground mt-2">
                        <Clock className="w-3 h-3" />
                        {notification.time}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-center">
                <Bell className="w-16 h-16 text-muted-foreground mb-4" />
                <h3 className="text-xl font-semibold mb-2">No Notifications</h3>
                <p className="text-muted-foreground">
                  You&apos;re all caught up! Real-time notifications will appear here.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper function to get relative time
function getRelativeTime(timestamp: number): string {
  const now = Date.now();
  const diff = now - timestamp;
  
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (seconds < 60) return "Just now";
  if (minutes < 60) return `${minutes} minute${minutes !== 1 ? "s" : ""} ago`;
  if (hours < 24) return `${hours} hour${hours !== 1 ? "s" : ""} ago`;
  return `${days} day${days !== 1 ? "s" : ""} ago`;
}
