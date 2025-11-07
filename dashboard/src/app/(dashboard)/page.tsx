// src/app/(dashboard)/page.tsx
import { Header } from "@/components/dashboard/header";
import { StatsCard } from "@/components/dashboard/stats-card";
import { ChildCard } from "@/components/dashboard/child-card";
import { api } from "@/lib/api";
import { Users, UserCheck, UserX, TrendingUp } from "lucide-react";

export default async function OverviewPage() {
  const children = await api.getChildren();
  
  const totalChildren = children.length;
  const foundChildren = children.filter((c) => c.status === "found").length;
  const missingChildren = children.filter((c) => c.status === "missing").length;
  const recentlyFound = children
    .filter((c) => c.status === "found")
    .slice(0, 3);

  return (
    <div className="flex flex-col h-screen">
      <Header
        title="Overview"
        description="Welcome to your Missing Person Management Dashboard"
      />
      
      <div className="flex-1 overflow-auto p-6 space-y-6">
        {/* Stats Grid */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <StatsCard
            title="Total Records"
            value={totalChildren}
            icon={Users}
            iconColor="text-blue-500"
            description="All registered cases"
          />
          <StatsCard
            title="Found"
            value={foundChildren}
            icon={UserCheck}
            iconColor="text-green-500"
            description="Successfully recovered"
          />
          <StatsCard
            title="Still Missing"
            value={missingChildren}
            icon={UserX}
            iconColor="text-red-500"
            description="Active cases"
          />
          <StatsCard
            title="Success Rate"
            value={`${totalChildren > 0 ? Math.round((foundChildren / totalChildren) * 100) : 0}%`}
            icon={TrendingUp}
            iconColor="text-purple-500"
            description="Recovery percentage"
          />
        </div>

        {/* Recently Found Section */}
        <div>
          <h2 className="text-2xl font-bold mb-4">Recently Found</h2>
          {recentlyFound.length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {recentlyFound.map((child) => (
                <ChildCard key={child.id} child={child} />
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              No recently found children
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
