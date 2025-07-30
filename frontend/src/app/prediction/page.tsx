"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Activity, AlertTriangle, CheckCircle, Info, ArrowLeft, RotateCcw } from "lucide-react"

interface AssessmentData {
  gender: string
  age: string
  patientHistory: string
  takesMedicines: string
  severity: string
  breathShortness: string
  visualChanges: string
  noseBleeding: string
  previouslyDiagnosed: string
  systolic: string
  diastolic: string
  controlledDiet: string
}

interface PredictionResult {
  disease: string
  confidence: number
  riskLevel: "low" | "moderate" | "high"
  description: string
  recommendations: string[]
}

export default function PredictionPage() {
  const router = useRouter()
  const [assessmentData, setAssessmentData] = useState<AssessmentData | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const data = localStorage.getItem("assessmentData")
    if (!data) {
      router.push("/assessment")
      return
    }

    const parsedData: AssessmentData = JSON.parse(data)
    setAssessmentData(parsedData)

    // Simulate ML prediction based on form data
    setTimeout(() => {
      const result = generatePrediction(parsedData)
      setPrediction(result)
      setIsLoading(false)
    }, 2000)
  }, [router])

  const generatePrediction = (data: AssessmentData): PredictionResult => {
    // Simple rule-based prediction simulation
    const age = Number.parseInt(data.age)
    const systolic = Number.parseInt(data.systolic)
    const hasHistory = data.patientHistory === "yes"
    const severity = data.severity
    const hasSymptoms = data.breathShortness === "yes" || data.visualChanges === "yes" || data.noseBleeding === "yes"

    let disease = "Healthy Skin"
    let confidence = 85
    let riskLevel: "low" | "moderate" | "high" = "low"
    let description = "Based on your assessment, your skin appears to be in good health."
    let recommendations = [
      "Maintain good skin hygiene",
      "Use sunscreen daily",
      "Stay hydrated",
      "Follow a balanced diet",
    ]

    if (severity === "severe" || (hasHistory && hasSymptoms)) {
      disease = "Psoriasis"
      confidence = 78
      riskLevel = "high"
      description = "Your symptoms suggest a possible autoimmune skin condition characterized by red, scaly patches."
      recommendations = [
        "Consult a dermatologist immediately",
        "Avoid known triggers",
        "Use prescribed topical treatments",
        "Consider phototherapy if recommended",
      ]
    } else if (severity === "moderate" || (age > 50 && systolic > 140)) {
      disease = "Eczema (Atopic Dermatitis)"
      confidence = 72
      riskLevel = "moderate"
      description = "Your assessment indicates possible eczema, a condition causing itchy, inflamed skin."
      recommendations = [
        "Schedule an appointment with a dermatologist",
        "Use gentle, fragrance-free moisturizers",
        "Avoid harsh soaps and detergents",
        "Identify and avoid personal triggers",
      ]
    } else if (hasSymptoms || data.previouslyDiagnosed === "yes") {
      disease = "Contact Dermatitis"
      confidence = 68
      riskLevel = "moderate"
      description = "Your symptoms may indicate contact dermatitis, likely caused by an allergic reaction or irritant."
      recommendations = [
        "Identify and avoid the triggering substance",
        "Apply cool compresses to affected areas",
        "Use over-the-counter antihistamines if needed",
        "See a doctor if symptoms persist",
      ]
    }

    return { disease, confidence, riskLevel, description, recommendations }
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "low":
        return "bg-green-100 text-green-800"
      case "moderate":
        return "bg-yellow-100 text-yellow-800"
      case "high":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case "low":
        return <CheckCircle className="h-4 w-4" />
      case "moderate":
        return <Info className="h-4 w-4" />
      case "high":
        return <AlertTriangle className="h-4 w-4" />
      default:
        return <Info className="h-4 w-4" />
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="pt-6">
            <div className="text-center space-y-4">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
              <h3 className="text-lg font-semibold">Analyzing Your Data</h3>
              <p className="text-gray-600">Our AI is processing your health information...</p>
              <Progress value={75} className="w-full" />
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!prediction || !assessmentData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="pt-6 text-center">
            <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">Error Loading Prediction</h3>
            <p className="text-gray-600 mb-4">Unable to load your assessment data.</p>
            <Link href="/assessment">
              <Button>Retake Assessment</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    )
  }

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
            <div className="flex items-center space-x-4">
              <Link href="/assessment" className="flex items-center space-x-2 text-gray-700 hover:text-blue-600">
                <RotateCcw className="h-4 w-4" />
                <span>Retake Assessment</span>
              </Link>
              <Link href="/" className="flex items-center space-x-2 text-gray-700 hover:text-blue-600">
                <ArrowLeft className="h-4 w-4" />
                <span>Home</span>
              </Link>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Your Prediction Results</h2>
            <p className="text-lg text-gray-600">Based on your health assessment, here's what our AI model predicts</p>
          </div>

          <div className="grid gap-6">
            {/* Main Prediction Card */}
            <Card className="border-2 border-blue-200">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-2xl">Predicted Condition</CardTitle>
                  <Badge className={`${getRiskColor(prediction.riskLevel)} flex items-center space-x-1`}>
                    {getRiskIcon(prediction.riskLevel)}
                    <span className="capitalize">{prediction.riskLevel} Risk</span>
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="text-3xl font-bold text-blue-600 mb-2">{prediction.disease}</h3>
                  <div className="flex items-center space-x-2 mb-4">
                    <span className="text-sm text-gray-600">Confidence Level:</span>
                    <Progress value={prediction.confidence} className="flex-1 max-w-xs" />
                    <span className="text-sm font-medium">{prediction.confidence}%</span>
                  </div>
                  <p className="text-gray-700 leading-relaxed">{prediction.description}</p>
                </div>
              </CardContent>
            </Card>

            {/* Recommendations Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Info className="h-5 w-5 text-blue-600" />
                  <span>Recommendations</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3">
                  {prediction.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start space-x-3">
                      <div className="bg-blue-100 text-blue-600 rounded-full w-6 h-6 flex items-center justify-center text-sm font-medium mt-0.5">
                        {index + 1}
                      </div>
                      <span className="text-gray-700">{rec}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {/* Assessment Summary */}
            <Card>
              <CardHeader>
                <CardTitle>Assessment Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-4 text-sm">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Age:</span>
                      <span className="font-medium">{assessmentData.age} years</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Gender:</span>
                      <span className="font-medium capitalize">{assessmentData.gender}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Severity:</span>
                      <span className="font-medium capitalize">{assessmentData.severity}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Medical History:</span>
                      <span className="font-medium capitalize">{assessmentData.patientHistory}</span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Blood Pressure:</span>
                      <span className="font-medium">
                        {assessmentData.systolic}/{assessmentData.diastolic}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Takes Medicines:</span>
                      <span className="font-medium capitalize">{assessmentData.takesMedicines}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Controlled Diet:</span>
                      <span className="font-medium capitalize">{assessmentData.controlledDiet}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Previously Diagnosed:</span>
                      <span className="font-medium capitalize">{assessmentData.previouslyDiagnosed}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Disclaimer */}
            <Card className="bg-yellow-50 border-yellow-200">
              <CardContent className="pt-6">
                <div className="flex items-start space-x-3">
                  <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5" />
                  <div>
                    <h4 className="font-semibold text-yellow-800 mb-2">Important Disclaimer</h4>
                    <p className="text-yellow-700 text-sm leading-relaxed">
                      This prediction is generated by an AI model and is for informational purposes only. It should not
                      be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult
                      with a qualified healthcare provider for proper medical evaluation and treatment.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/assessment">
                <Button variant="outline" size="lg" className="w-full sm:w-auto bg-transparent">
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Take New Assessment
                </Button>
              </Link>
              <Link href="/">
                <Button size="lg" className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700">
                  Back to Home
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
