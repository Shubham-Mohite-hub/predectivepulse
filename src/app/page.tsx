import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Activity, Brain, Shield, Users } from "lucide-react"

export default function HomePage() {
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
            <nav className="hidden md:flex space-x-6">
              <Link href="/" className="text-gray-700 hover:text-blue-600">
                Home
              </Link>
              <Link href="/assessment" className="text-gray-700 hover:text-blue-600">
                Assessment
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20">
        <div className="container mx-auto px-4 text-center">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-5xl font-bold text-gray-900 mb-6">Advanced Skin Disease Prediction</h2>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed">
              Welcome to Predictive Pulse - your trusted partner in early skin disease detection. Using cutting-edge
              machine learning technology, we analyze your health data to provide accurate predictions and help you take
              proactive steps towards better skin health.
            </p>
            <Link href="/assesment">
              <Button size="lg" className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 text-lg">
                Start Your Assessment
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-white">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">What We Do</h3>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our advanced AI system analyzes multiple health indicators to predict potential skin diseases, helping you
              make informed decisions about your health.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="text-center p-6 hover:shadow-lg transition-shadow">
              <CardContent className="pt-6">
                <Brain className="h-12 w-12 text-blue-600 mx-auto mb-4" />
                <h4 className="text-xl font-semibold mb-3">AI-Powered Analysis</h4>
                <p className="text-gray-600">
                  Our machine learning algorithms analyze your health data using advanced pattern recognition to
                  identify potential skin conditions.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center p-6 hover:shadow-lg transition-shadow">
              <CardContent className="pt-6">
                <Shield className="h-12 w-12 text-green-600 mx-auto mb-4" />
                <h4 className="text-xl font-semibold mb-3">Early Detection</h4>
                <p className="text-gray-600">
                  Get early warnings about potential skin diseases, allowing you to seek medical attention before
                  conditions worsen.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center p-6 hover:shadow-lg transition-shadow">
              <CardContent className="pt-6">
                <Users className="h-12 w-12 text-purple-600 mx-auto mb-4" />
                <h4 className="text-xl font-semibold mb-3">Personalized Results</h4>
                <p className="text-gray-600">
                  Receive personalized predictions based on your unique health profile, medical history, and current
                  symptoms.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">How It Works</h3>
            <p className="text-lg text-gray-600">Simple steps to get your skin health prediction</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-blue-600 text-white rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4 text-xl font-bold">
                1
              </div>
              <h4 className="text-xl font-semibold mb-3">Complete Assessment</h4>
              <p className="text-gray-600">
                Fill out our comprehensive health questionnaire with your medical history and current symptoms.
              </p>
            </div>

            <div className="text-center">
              <div className="bg-green-600 text-white rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4 text-xl font-bold">
                2
              </div>
              <h4 className="text-xl font-semibold mb-3">AI Analysis</h4>
              <p className="text-gray-600">
                Our machine learning model processes your data and compares it against thousands of medical cases.
              </p>
            </div>

            <div className="text-center">
              <div className="bg-purple-600 text-white rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4 text-xl font-bold">
                3
              </div>
              <h4 className="text-xl font-semibold mb-3">Get Results</h4>
              <p className="text-gray-600">
                Receive your personalized prediction report with recommendations for next steps.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-blue-600 text-white">
        <div className="container mx-auto px-4 text-center">
          <h3 className="text-3xl font-bold mb-4">Ready to Check Your Skin Health?</h3>
          <p className="text-xl mb-8 opacity-90">
            Take our quick assessment and get AI-powered insights about your skin health in minutes.
          </p>
          <Link href="/assessment">
            <Button size="lg" variant="secondary" className="px-8 py-3 text-lg">
              Start Assessment Now
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8">
        <div className="container mx-auto px-4 text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <Activity className="h-6 w-6" />
            <span className="text-lg font-semibold">Predictive Pulse</span>
          </div>
          <p className="text-gray-400">Advanced AI-powered skin disease prediction for better health outcomes.</p>
          <p className="text-sm text-gray-500 mt-4">
            © 2024 Predictive Pulse. This tool is for informational purposes only and should not replace professional
            medical advice.
          </p>
        </div>
      </footer>
    </div>
  )
}
