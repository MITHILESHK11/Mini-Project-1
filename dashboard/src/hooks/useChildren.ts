// src/hooks/useChildren.ts
"use client";

import { useState, useEffect, useCallback } from "react";
import { Child } from "@/lib/types";
import { api } from "@/lib/api";

export function useChildren() {
  const [children, setChildren] = useState<Child[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchChildren = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.getChildren();
      setChildren(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchChildren();
  }, [fetchChildren]);

  return { children, loading, error, refetch: fetchChildren };
}
