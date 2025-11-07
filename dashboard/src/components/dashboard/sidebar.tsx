// src/components/dashboard/sidebar.tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  Home,
  UserPlus,
  CheckCircle,
  Database,
  Bell,
  LogOut,
  Shield,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useRouter } from "next/navigation";

const routes = [
  {
    label: "Overview",
    icon: Home,
    href: "/",
    color: "text-sky-500",
  },
  {
    label: "Add Child",
    icon: UserPlus,
    href: "/add-child",
    color: "text-violet-500",
  },
  {
    label: "Recently Found",
    icon: CheckCircle,
    href: "/recently-found",
    color: "text-green-500",
  },
  {
    label: "All Records",
    icon: Database,
    href: "/all-records",
    color: "text-orange-500",
  },
  {
    label: "Notifications",
    icon: Bell,
    href: "/notifications",
    color: "text-pink-500",
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();

  const handleLogout = async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    router.push("/login");
    router.refresh();
  };

  return (
    <div className="h-full flex flex-col bg-card border-r">
      {/* Logo */}
      <div className="p-6">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
            <Shield className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="text-xl font-bold">MissingPerson</span>
        </Link>
      </div>

      <Separator />

      {/* Navigation */}
      <div className="flex-1 p-4 space-y-1">
        {routes.map((route) => (
          <Link key={route.href} href={route.href}>
            <div
              className={cn(
                "group flex items-center gap-3 px-3 py-3 rounded-lg transition-all hover:bg-accent",
                pathname === route.href
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground"
              )}
            >
              <route.icon className={cn("w-5 h-5", route.color)} />
              <span className="font-medium">{route.label}</span>
            </div>
          </Link>
        ))}
      </div>

      <Separator />

      {/* Logout */}
      <div className="p-4">
        <Button
          variant="ghost"
          className="w-full justify-start"
          onClick={handleLogout}
        >
          <LogOut className="w-5 h-5 mr-3" />
          Logout
        </Button>
      </div>
    </div>
  );
}
