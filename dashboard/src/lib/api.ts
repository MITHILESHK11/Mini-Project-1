// src/lib/api.ts
import { Child, AddChildData } from "./types";

const API_URL = "http://127.0.0.1:8000";

export const api = {
  async getChildren(): Promise<Child[]> {
    const response = await fetch(`${API_URL}/get_children/`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error("Failed to fetch children");
    return response.json();
  },

  async addChild(data: AddChildData): Promise<{ message: string }> {
    const formData = new FormData();
    formData.append("name", data.name);
    formData.append("age", data.age.toString());
    formData.append("contact", data.contact);
    formData.append("image", data.image);

    const response = await fetch(`${API_URL}/add_child/`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Failed to add child");
    return response.json();
  },
};
