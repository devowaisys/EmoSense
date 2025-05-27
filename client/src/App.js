import { BrowserRouter, Route, Routes } from "react-router-dom";
import UserContextProvider from "./UserStore";
import AnalysisContextProvider from "./PreRecordedStore";
import Index from "./pages/Index";
import AboutUs from "./pages/AboutUs";
import PrivacyPolicy from "./pages/PrivacyPolicy";
import Home from "./pages/Home";
import PreRecorded from "./pages/PreRecorded";
import Realtime from "./pages/Realtime";
import History from "./pages/History";
import ProtectedRoute from "./pages/ProtectedRoute";

function App() {
  return (
      <UserContextProvider>
        <AnalysisContextProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/about" element={<AboutUs />} />
              <Route path="/privacy" element={<PrivacyPolicy />} />
              <Route
                  path="/home"
                  element={
                    <ProtectedRoute>
                      <Home />
                    </ProtectedRoute>
                  }
              />
              <Route
                  path="/realtime/:email/:fullname/:contact"
                  element={
                    <ProtectedRoute>
                      <Realtime />
                    </ProtectedRoute>
                  }
              />
              <Route
                  path="/prerecorded"
                  element={
                    <ProtectedRoute>
                      <PreRecorded />
                    </ProtectedRoute>
                  }
              />
              <Route
                  path="/history"
                  element={
                    <ProtectedRoute>
                      <History />
                    </ProtectedRoute>
                  }
              />
            </Routes>
          </BrowserRouter>
        </AnalysisContextProvider>
      </UserContextProvider>
  );
}

export default App;
