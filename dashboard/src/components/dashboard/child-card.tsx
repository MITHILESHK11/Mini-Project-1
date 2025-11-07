// src/components/dashboard/child-card.tsx
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Child } from "@/lib/types";
import Image from "next/image";
import { Clock, Phone, User } from "lucide-react";

const API_URL = "http://127.0.0.1:8000";

interface ChildCardProps {
  child: Child;
}

export function ChildCard({ child }: ChildCardProps) {
  return (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow">
      <div className="relative w-full h-48 bg-muted">
        <img
          src={`${API_URL}/${child.image}`}
          alt={child.name}
          fill
          className="object-cover"
        />
        <div className="absolute top-2 right-2">
          <Badge
            variant={child.status === "found" ? "default" : "destructive"}
            className={
              child.status === "found"
                ? "bg-green-500 hover:bg-green-600"
                : ""
            }
          >
            {child.status === "found" ? "Found" : "Missing"}
          </Badge>
        </div>
      </div>
      <CardContent className="p-4 space-y-2">
        <h3 className="font-semibold text-lg flex items-center gap-2">
          <User className="w-4 h-4 text-muted-foreground" />
          {child.name}
        </h3>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Clock className="w-4 h-4" />
          <span>{child.age} years old</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Phone className="w-4 h-4" />
          <span>{child.contact}</span>
        </div>
      </CardContent>
    </Card>
  );
}
