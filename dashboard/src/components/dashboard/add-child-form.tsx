// src/components/dashboard/add-child-form.tsx
"use client";

import { useState, FormEvent } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import { Upload } from "lucide-react";
import { useRouter } from "next/navigation";

export function AddChildForm() {
  const [formData, setFormData] = useState({
    name: "",
    age: 0,
    contact: "",
  });
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    
    if (!image) {
      toast.error("Please upload an image");
      return;
    }

    setLoading(true);

    try {
      const result = await api.addChild({ ...formData, image });
      toast.success("Child record added successfully!", {
        description: `${formData.name} has been added to the database.`,
      });
      
      // Reset form
      setFormData({ name: "", age: 0, contact: "" });
      setImage(null);
      setPreview(null);
      
      // Reset file input
      const fileInput = document.getElementById("image-upload") as HTMLInputElement;
      if (fileInput) fileInput.value = "";

      // Refresh the page to show updated data
      router.refresh();
    } catch (error) {
      toast.error("Failed to add child", {
        description: error instanceof Error ? error.message : "Please try again",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="max-w-2xl">
      <CardHeader>
        <CardTitle>Add New Child Record</CardTitle>
        <CardDescription>
          Fill in the details below to register a new missing person case
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Image Upload */}
          <div className="space-y-2">
            <Label htmlFor="image-upload">Photo</Label>
            <div className="flex items-start gap-4">
              {preview ? (
                <div className="relative w-32 h-32 rounded-lg overflow-hidden border">
                  <img
                    src={preview}
                    alt="Preview"
                    className="w-full h-full object-cover"
                  />
                </div>
              ) : (
                <div className="w-32 h-32 rounded-lg border-2 border-dashed border-muted-foreground/25 flex items-center justify-center">
                  <Upload className="w-8 h-8 text-muted-foreground" />
                </div>
              )}
              <div className="flex-1">
                <Input
                  id="image-upload"
                  type="file"
                  accept="image/jpg,image/jpeg,image/png"
                  onChange={handleImageChange}
                  required
                />
                <p className="text-xs text-muted-foreground mt-2">
                  Upload a clear photo (JPG, PNG). Max 5MB.
                </p>
              </div>
            </div>
          </div>

          {/* Name */}
          <div className="space-y-2">
            <Label htmlFor="name">Full Name</Label>
            <Input
              id="name"
              type="text"
              placeholder="Enter full name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
            />
          </div>

          {/* Age */}
          <div className="space-y-2">
            <Label htmlFor="age">Age</Label>
            <Input
              id="age"
              type="number"
              min="0"
              max="120"
              placeholder="Enter age"
              value={formData.age || ""}
              onChange={(e) =>
                setFormData({ ...formData, age: parseInt(e.target.value) || 0 })
              }
              required
            />
          </div>

          {/* Contact */}
          <div className="space-y-2">
            <Label htmlFor="contact">Contact Information</Label>
            <Input
              id="contact"
              type="text"
              placeholder="Phone number or email"
              value={formData.contact}
              onChange={(e) =>
                setFormData({ ...formData, contact: e.target.value })
              }
              required
            />
          </div>

          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? "Adding..." : "Add Record"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
