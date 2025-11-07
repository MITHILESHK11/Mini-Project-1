// src/components/AllRecords.tsx
import { Child } from "@/lib/types"; 

const API_URL = "http://localhost:8000";

interface AllRecordsProps {
  children: Child[];
  loading: boolean;
}

export default function AllRecords({ children, loading }: AllRecordsProps) {
  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">All Database Records</h2>
        <p className="text-gray-500">Loading...</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold mb-4">All Database Records</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {children.map((child) => (
          <div
            key={child.id}
            className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow"
          >
            <div className="relative w-full h-48">
              <img
                src={`${API_URL}/${child.image}`}  
                alt={child.name}
                fill
                className="object-cover"
              />
            </div>
            
            <div className="p-4">
              <h3 className="font-semibold text-lg mb-2">{child.name}</h3>
              
              <div className="space-y-1 text-sm text-gray-700">
                <p><strong>Age:</strong> {child.age}</p>
                <p><strong>Contact:</strong> {child.contact}</p>
                <p>
                  <strong>Status:</strong>{" "}
                  {child.status === "found" ? (
                    <span className="text-green-600 font-medium">🟢 Found</span>
                  ) : (
                    <span className="text-red-600 font-medium">🔴 Missing</span>
                  )}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
