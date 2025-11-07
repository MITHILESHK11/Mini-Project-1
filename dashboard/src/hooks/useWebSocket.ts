// src/hooks/useWebSocket.ts
"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { WebSocketNotification } from "@/lib/types";

const WS_URL = "ws://127.0.0.1:8000/ws/notifications";

export function useWebSocket() {
  const [notification, setNotification] = useState<WebSocketNotification | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log("✅ WebSocket connected");
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data: WebSocketNotification = JSON.parse(event.data);
          setNotification(data);
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
        setIsConnected(false);
      };

      ws.onclose = () => {
        console.log("🔌 WebSocket disconnected");
        setIsConnected(false);
        // Auto-reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log("🔄 Reconnecting...");
          connect();
        }, 3000);
      };

      wsRef.current = ws;
    } catch (error) {
      console.error("WebSocket connection failed:", error);
    }
  }, []);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const clearNotification = useCallback(() => {
    setNotification(null);
  }, []);

  return { notification, isConnected, clearNotification };
}
