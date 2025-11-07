// src/lib/auth.ts
import { cookies } from "next/headers";

const ADMIN_USERNAME = "admin";
const ADMIN_PASSWORD = "admin123"; // Change this in production!

export async function verifyAuth() {
  const cookieStore = await cookies();
  const token = cookieStore.get("auth-token");
  return token?.value === "authenticated";
}

export function validateCredentials(username: string, password: string) {
  return username === ADMIN_USERNAME && password === ADMIN_PASSWORD;
}
