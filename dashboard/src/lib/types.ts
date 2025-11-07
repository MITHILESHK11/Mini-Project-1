// src/lib/types.ts
export interface Child {
  id: string;
  name: string;
  age: number;
  contact: string;
  image: string;
  status: "missing" | "found";
}

export interface WebSocketNotification {
  event: string;
  child?: Child;
  message?: string;
}

export interface AddChildData {
  name: string;
  age: number;
  contact: string;
  image: File;
}
