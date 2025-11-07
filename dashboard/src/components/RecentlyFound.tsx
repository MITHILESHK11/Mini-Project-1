// src/components/RecentlyFound.tsx
"use client";

import { Child } from "@/lib/types"; 

const API_URL = "http://localhost:8000";

interface RecentlyFoundProps {
  children: Child[];
}

export default function RecentlyFound({ children }: RecentlyFoundProps) {
  const recentlyFound = children.filter((child) => child.status === "found");

  if (recentlyFound.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">Recently Found</h2>
        <p className="text-gray-500">No recently found children</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold mb-4">Recently Found</h2>
      
      <div className="space-y-4">
        {recentlyFound.map((child) => (
          <div
            key={child.id}
            className="bg-green-50 border-l-4 border-green-500 p-4 rounded flex items-start gap-4"
          >
            <div className="relative w-24 h-24 flex-shrink-0">
              <img
                src={`${API_URL}/${child.image}`} 
                alt={child.name}
                fill
                className="object-cover rounded"
              />
            </div>
            
            <div className="flex-1">
              <p className="font-semibold text-green-800">
                🟢 Found: {child.name} (ID: {child.id})
              </p>
              <p className="text-sm text-gray-700 mt-1">
                <strong>Contact:</strong> {child.contact}
              </p>
              <p className="text-sm text-gray-700">
                <strong>Age:</strong> {child.age}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
