import { createContext, useState, useCallback } from "react";

export const AnalysisContext = createContext({
  analysisResult: null,
  patientInfo: null,
  storeAnalysisResult: () => {},
  resetAnalysisResult: () => {},
  updateAnalysisResult: () => {},
  savePatientInfo: () => {},
  resetPatientInfo: () => {},
});

export default function AnalysisContextProvider({ children }) {
  const [analysisResult, setAnalysisResult] = useState(() => {
    const savedResult = localStorage.getItem("analysisResult");
    return savedResult ? JSON.parse(savedResult) : null;
  });

  const [patientInfo, setPatientInfo] = useState(() => {
    const savedInfo = localStorage.getItem("patientInfo");
    return savedInfo ? JSON.parse(savedInfo) : null;
  });

  const storeAnalysisResult = useCallback((result) => {
    setAnalysisResult(result);
    localStorage.setItem("analysisResult", JSON.stringify(result));
  }, []);

  const resetAnalysisResult = useCallback(() => {
    setAnalysisResult(null);
    localStorage.removeItem("analysisResult");
  }, []);

  const updateAnalysisResult = useCallback((updatedFields) => {
    setAnalysisResult((prevResult) => {
      const newResult = { ...prevResult, ...updatedFields };
      localStorage.setItem("analysisResult", JSON.stringify(newResult));
      return newResult;
    });
  }, []);

  const savePatientInfo = useCallback((patient) => {
    setPatientInfo(patient);
    localStorage.setItem("patientInfo", JSON.stringify(patient));
  }, []);

  const resetPatientInfo = useCallback(() => {
    setPatientInfo(null);
    localStorage.removeItem("patientInfo");
  }, []);

  const contextValue = {
    analysisResult,
    patientInfo,
    storeAnalysisResult,
    resetAnalysisResult,
    updateAnalysisResult,
    savePatientInfo,
    resetPatientInfo,
  };

  return (
    <AnalysisContext.Provider value={contextValue}>
      {children}
    </AnalysisContext.Provider>
  );
}
