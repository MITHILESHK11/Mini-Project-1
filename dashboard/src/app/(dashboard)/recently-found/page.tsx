// src/app/(dashboard)/recently-found/page.tsx
import { Header } from "@/components/dashboard/header";
import { ChildCard } from "@/components/dashboard/child-card";
import { api } from "@/lib/api";
import { CheckCircle } from "lucide-react";

export default async function RecentlyFoundPage() {
  const children = await api.getChildren();
  const recentlyFound = children.filter((c) => c.status === "found");

  return (
    <div className="flex flex-col h-screen">
      <Header
        title="Recently Found"
        description={`${recentlyFound.length} children have been successfully recovered`}
      />
      
      <div className="flex-1 overflow-auto p-6">
        {recentlyFound.length > 0 ? (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {recentlyFound.map((child) => (
              <ChildCard key={child.id} child={child} />
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <CheckCircle className="w-16 h-16 text-muted-foreground mb-4" />
            <h3 className="text-xl font-semibold mb-2">No Found Records Yet</h3>
            <p className="text-muted-foreground">
              When children are found, they will appear here
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
