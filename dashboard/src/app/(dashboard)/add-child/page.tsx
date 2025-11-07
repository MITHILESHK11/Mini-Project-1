// src/app/(dashboard)/add-child/page.tsx
import { Header } from "@/components/dashboard/header";
import { AddChildForm } from "@/components/dashboard/add-child-form";

export default function AddChildPage() {
  return (
    <div className="flex flex-col h-screen">
      <Header
        title="Add New Record"
        description="Register a new missing person case"
      />
      
      <div className="flex-1 overflow-auto p-6">
        <AddChildForm />
      </div>
    </div>
  );
}
