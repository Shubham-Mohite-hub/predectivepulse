"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Activity, ArrowLeft } from "lucide-react"

export default function AssessmentPage() {
  const router = useRouter()
  const [formData, setFormData] = useState({
    gender: "",
    age: "",
    patientHistory: "",
    takesMedicines: "",
    severity: "",
    breathShortness: "",
    visualChanges: "",
    noseBleeding: "",
    previouslyDiagnosed: "",
    systolic: "",
    diastolic: "",
    controlledDiet: "",
  })

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Store form data in localStorage to pass to prediction page
    localStorage.setItem("assessmentData", JSON.stringify(formData))
    router.push("/prediction")
  }

  const isFormValid = Object.values(formData).every((value) => value !== "")

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Activity className="h-8 w-8 text-blue-600" />
              <h1 className="text-2xl font-bold text-gray-900">Predictive Pulse</h1>
            </div>
            <Link href="/" className="flex items-center space-x-2 text-gray-700 hover:text-blue-600">
              <ArrowLeft className="h-4 w-4" />
              <span>Back to Home</span>
            </Link>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Health Assessment</h2>
            <p className="text-lg text-gray-600">
              Please provide accurate information about your health condition. This data will be used to predict
              potential skin diseases.
            </p>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Patient Information</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Gender */}
                <div className="space-y-3">
                  <Label className="text-base font-medium">Gender</Label>
                  <RadioGroup value={formData.gender} onValueChange={(value) => handleInputChange("gender", value)}>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="male" id="male" />
                      <Label htmlFor="male">Male</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="female" id="female" />
                      <Label htmlFor="female">Female</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="other" id="other" />
                      <Label htmlFor="other">Other</Label>
                    </div>
                  </RadioGroup>
                </div>

                {/* Age */}
                <div className="space-y-2">
                  <Label htmlFor="age" className="text-base font-medium">
                    Age
                  </Label>
                  <Input
                    id="age"
                    type="number"
                    placeholder="Enter your age"
                    value={formData.age}
                    onChange={(e) => handleInputChange("age", e.target.value)}
                    min="1"
                    max="120"
                  />
                </div>

                {/* Patient History */}
                <div className="space-y-3">
                  <Label className="text-base font-medium">Do you have any significant medical history?</Label>
                  <RadioGroup
                    value={formData.patientHistory}
                    onValueChange={(value) => handleInputChange("patientHistory", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="yes" id="history-yes" />
                      <Label htmlFor="history-yes">Yes</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no" id="history-no" />
                      <Label htmlFor="history-no">No</Label>
                    </div>
                  </RadioGroup>
                </div>

                {/* Takes Medicines */}
                <div className="space-y-3">
                  <Label className="text-base font-medium">Are you currently taking any medications?</Label>
                  <RadioGroup
                    value={formData.takesMedicines}
                    onValueChange={(value) => handleInputChange("takesMedicines", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="yes" id="meds-yes" />
                      <Label htmlFor="meds-yes">Yes</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no" id="meds-no" />
                      <Label htmlFor="meds-no">No</Label>
                    </div>
                  </RadioGroup>
                </div>

                {/* Severity */}
                <div className="space-y-2">
                  <Label className="text-base font-medium">How would you rate the severity of your symptoms?</Label>
                  <Select value={formData.severity} onValueChange={(value) => handleInputChange("severity", value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select severity level" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="mild">Mild</SelectItem>
                      <SelectItem value="moderate">Moderate</SelectItem>
                      <SelectItem value="severe">Severe</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Breath Shortness */}
                <div className="space-y-3">
                  <Label className="text-base font-medium">Do you experience shortness of breath?</Label>
                  <RadioGroup
                    value={formData.breathShortness}
                    onValueChange={(value) => handleInputChange("breathShortness", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="yes" id="breath-yes" />
                      <Label htmlFor="breath-yes">Yes</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no" id="breath-no" />
                      <Label htmlFor="breath-no">No</Label>
                    </div>
                  </RadioGroup>
                </div>

                {/* Visual Changes */}
                <div className="space-y-3">
                  <Label className="text-base font-medium">Have you noticed any visual changes?</Label>
                  <RadioGroup
                    value={formData.visualChanges}
                    onValueChange={(value) => handleInputChange("visualChanges", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="yes" id="visual-yes" />
                      <Label htmlFor="visual-yes">Yes</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no" id="visual-no" />
                      <Label htmlFor="visual-no">No</Label>
                    </div>
                  </RadioGroup>
                </div>

                {/* Nose Bleeding */}
                <div className="space-y-3">
                  <Label className="text-base font-medium">Do you experience nose bleeding?</Label>
                  <RadioGroup
                    value={formData.noseBleeding}
                    onValueChange={(value) => handleInputChange("noseBleeding", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="yes" id="nose-yes" />
                      <Label htmlFor="nose-yes">Yes</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no" id="nose-no" />
                      <Label htmlFor="nose-no">No</Label>
                    </div>
                  </RadioGroup>
                </div>

                {/* Previously Diagnosed */}
                <div className="space-y-3">
                  <Label className="text-base font-medium">Have you been diagnosed with a skin condition before?</Label>
                  <RadioGroup
                    value={formData.previouslyDiagnosed}
                    onValueChange={(value) => handleInputChange("previouslyDiagnosed", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="yes" id="diagnosed-yes" />
                      <Label htmlFor="diagnosed-yes">Yes</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no" id="diagnosed-no" />
                      <Label htmlFor="diagnosed-no">No</Label>
                    </div>
                  </RadioGroup>
                </div>

                {/* Blood Pressure */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="systolic" className="text-base font-medium">
                      Systolic Blood Pressure
                    </Label>
                    <Input
                      id="systolic"
                      type="number"
                      placeholder="e.g., 120"
                      value={formData.systolic}
                      onChange={(e) => handleInputChange("systolic", e.target.value)}
                      min="70"
                      max="250"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="diastolic" className="text-base font-medium">
                      Diastolic Blood Pressure
                    </Label>
                    <Input
                      id="diastolic"
                      type="number"
                      placeholder="e.g., 80"
                      value={formData.diastolic}
                      onChange={(e) => handleInputChange("diastolic", e.target.value)}
                      min="40"
                      max="150"
                    />
                  </div>
                </div>

                {/* Controlled Diet */}
                <div className="space-y-3">
                  <Label className="text-base font-medium">Do you follow a controlled diet?</Label>
                  <RadioGroup
                    value={formData.controlledDiet}
                    onValueChange={(value) => handleInputChange("controlledDiet", value)}
                  >
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="yes" id="diet-yes" />
                      <Label htmlFor="diet-yes">Yes</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="no" id="diet-no" />
                      <Label htmlFor="diet-no">No</Label>
                    </div>
                  </RadioGroup>
                </div>

                <Button
                  type="submit"
                  className="w-full bg-blue-600 hover:bg-blue-700"
                  size="lg"
                  disabled={!isFormValid}
                >
                  Get Prediction
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
