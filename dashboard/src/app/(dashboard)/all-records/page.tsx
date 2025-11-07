// src/app/(dashboard)/all-records/page.tsx
import { Header } from "@/components/dashboard/header";
import { ChildCard } from "@/components/dashboard/child-card";
import { api } from "@/lib/api";
import { Database } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default async function AllRecordsPage() {
  const children = await api.getChildren();
  const missing = children.filter((c) => c.status === "missing");
  const found = children.filter((c) => c.status === "found");

  return (
    <div className="flex flex-col h-screen">
      <Header
        title="All Records"
        description={`Total: ${children.length} records in database`}
      />
      
      <div className="flex-1 overflow-auto p-6">
        <Tabs defaultValue="all" className="w-full">
          <TabsList>
            <TabsTrigger value="all">All ({children.length})</TabsTrigger>
            <TabsTrigger value="missing">Missing ({missing.length})</TabsTrigger>
            <TabsTrigger value="found">Found ({found.length})</TabsTrigger>
          </TabsList>

          <TabsContent value="all" className="mt-6">
            {children.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {children.map((child) => (
                  <ChildCard key={child.id} child={child} />
                ))}
              </div>
            ) : (
              <EmptyState />
            )}
          </TabsContent>

          <TabsContent value="missing" className="mt-6">
            {missing.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {missing.map((child) => (
                  <ChildCard key={child.id} child={child} />
                ))}
              </div>
            ) : (
              <EmptyState message="No missing records" />
            )}
          </TabsContent>

          <TabsContent value="found" className="mt-6">
            {found.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {found.map((child) => (
                  <ChildCard key={child.id} child={child} />
                ))}
              </div>
            ) : (
              <EmptyState message="No found records" />
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

function EmptyState({ message = "No records found" }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 text-center">
      <Database className="w-16 h-16 text-muted-foreground mb-4" />
      <h3 className="text-xl font-semibold mb-2">{message}</h3>
      <p className="text-muted-foreground">
        Start by adding a new child record
      </p>
    </div>
  );
}
