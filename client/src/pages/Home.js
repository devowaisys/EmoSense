import Header from "../components/Header";
import cassette from "../assets/icons/casette.png";
import mic from "../assets/icons/microphone.png";
import Button from "../components/Button";
import { useState, useContext } from "react";
import axios from "axios";
import { AnalysisContext } from "../PreRecordedStore";
import { useNavigate } from "react-router-dom";
import PatientDetailsPopup from "../components/PatientDetailsPopup";
import { UserContext } from "../UserStore";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [err, setError] = useState("");
  const [message, setMessage] = useState("");
  const { storeAnalysisResult, savePatientInfo } = useContext(AnalysisContext);
  const [patientDetailsPopupIsVisible, setpatientDetailsPopupIsVisible] =
    useState(false);
  const [patientFullname, setPatientFullName] = useState("");
  const [patientEmail, setPatientEmail] = useState("");
  const [patientContact, setPatientContact] = useState("");
  const [currentMode, setCurrentMode] = useState(""); // Track current mode
  const { user } = useContext(UserContext);
  const navigate = useNavigate();

  function toggleResetPopup() {
    setpatientDetailsPopupIsVisible(false);
  }

  function handlePatientDetailsPopupIsVisible(mode) {
    setCurrentMode(mode);
    setpatientDetailsPopupIsVisible(true);
  }

  function handleFileChange(event) {
    setError("");
    setMessage("");
    setSelectedFile(event.target.files[0]);
  }

  async function handleSubmit() {
    setError("");
    setMessage("");

    if (!selectedFile) {
      setError("No file selected. Please choose a file to analyze.");
      return;
    }

    const formData = new FormData();
    formData.append("audio", selectedFile);
    formData.append("patientFullname", patientFullname);
    formData.append("patientEmail", patientEmail);
    formData.append("patientContact", patientContact);
    formData.append("therapistId", user.id);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/analyze/file",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          withCredentials: true,
          timeout: 10000,
        }
      );

      const dataJson = response.data;

      if (dataJson.error) {
        setError(dataJson.error);
      } else {
        setMessage("Analysis successful!");
        console.log("Analysis Result:", dataJson);
        navigate("/prerecorded");

        // Extract required fields from the response and store them in the context
        const analysisData = {
          average_confidence:
            dataJson.summary.emotion_analysis.average_confidences,
          dominant_emotion: dataJson.summary.emotion_analysis.dominant_emotions,
          duration: dataJson.summary.session_info.duration_minutes,
          description: dataJson.summary.summary_description,
        };
        storeAnalysisResult(analysisData);
        savePatientInfo({
          name: patientFullname,
          email: patientEmail,
          contact: patientContact,
        });
        console.log(analysisData);
      }
    } catch (err) {
      setError(
        err.response?.data?.error || "An error occurred during submission."
      );
    }
  }

  function handleSaveSuccess(mode) {
    if (mode === "pre-recorded") {
      document.getElementById("file-input").click();
    } else if (mode === "realtime") {
      const encodedEmail = encodeURIComponent(patientEmail);
      const encodedFullname = encodeURIComponent(patientFullname);
      const encodedContact = encodeURIComponent(patientContact);

      navigate(
        `/realtime/${encodedEmail}/${encodedFullname}/${encodedContact}`
      );
    }
  }

  return (
    <>
      {patientDetailsPopupIsVisible && (
        <PatientDetailsPopup
          fullname={patientFullname}
          email={patientEmail}
          contact={patientContact}
          setFullName={setPatientFullName}
          setEmail={setPatientEmail}
          setContact={setPatientContact}
          toggleResetPopup={toggleResetPopup}
          mode={currentMode}
          onSaveSuccess={handleSaveSuccess}
        />
      )}
      <Header />
      <div className="main-container">
        <main>
          <div className="left-container">
            <div className="card">
              <h3 style={{ margin: 0 }}>Analyse Pre Recorded Audio</h3>
              <p>
                Unlock insights hidden in your pre-recorded audio files with
                just a click!
              </p>
              <div className="button-container-simple">
                <Button
                  width={200}
                  text="Import File"
                  marginTop={10}
                  onClick={() =>
                    handlePatientDetailsPopupIsVisible("pre-recorded")
                  }
                />
                <input
                  onChange={handleFileChange}
                  type="file"
                  id="file-input"
                  style={{ display: "none" }}
                  accept="audio/*"
                />
                {selectedFile && (
                  <Button
                    width={200}
                    text="Start Analysis"
                    marginTop={10}
                    onClick={handleSubmit}
                  />
                )}
              </div>

              {err && (
                <span
                  className="err-msg"
                  style={{ textAlign: "left", marginTop: "1rem" }}
                >
                  Error: {err}
                </span>
              )}
              {message && (
                <span
                  className="success-msg"
                  style={{ textAlign: "left", marginTop: "1rem" }}
                >
                  {message}
                </span>
              )}
              {selectedFile && (
                <span className="file-name">
                  Uploaded File: {selectedFile.name}
                </span>
              )}
            </div>
          </div>
          <div className="right-container">
            <img
              src={cassette}
              alt="Audio"
              style={{ width: "17rem", height: "17rem" }}
            />
          </div>
        </main>

        <main>
          <div className="left-container">
            <img
              src={mic}
              alt="Audio"
              style={{ width: "14rem", height: "14rem" }}
            />
          </div>
          <div className="right-container">
            <div className="card">
              <h3 style={{ margin: 0 }}>Analyse Realtime Audio</h3>
              <p>
                Experience real-time emotion analysis as you speak, live and
                uninterrupted!
              </p>
              <Button
                width={200}
                text="Start Recording"
                marginTop={10}
                onClick={() => handlePatientDetailsPopupIsVisible("realtime")}
              />
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
