import { useEffect, useState, useContext } from "react";
import axios from "axios";
import Header from "../components/Header";
import HistoryItem from "../components/HistoryItem";
import { UserContext } from "../UserStore";
import NoData from "../assets/icons/no-data.png";

export default function History() {
  const { user, accessToken } = useContext(UserContext);
  const [err, setError] = useState("");
  const [analyses, setAnalyses] = useState([]);
  const [filteredPatients, setFilteredPatients] = useState([]);
  const [patientEmail, setPatientEmail] = useState("");

  useEffect(() => {
    const fetchAnalyses = async () => {
      try {
        const response = await axios.get(
          `http://127.0.0.1:5000/api/get_analysis_by_therapist_id/${user.id}`,
          {
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${accessToken}`,
            },
            timeout: 10000,
          }
        );

        if (response.data.success === true) {
          setAnalyses(response.data.analysis_results);
          setFilteredPatients(response.data.analysis_results); // Set initial full list
          setError("");
        } else {
          setError(response.data.message || "Fetch failed.");
        }
      } catch (err) {
        setError(err.response?.data?.message || err.message);
      }
    };

    if (user && user.id) {
      fetchAnalyses();
    }
  }, [user, accessToken]);

  const handleSearchChange = (e) => {
    const value = e.target.value;
    setPatientEmail(value);

    const filtered = analyses.filter((analysis) =>
      analysis.patient_email.toLowerCase().includes(value.toLowerCase())
    );
    setFilteredPatients(filtered);
  };

  return (
    <>
      <Header />
      <div className="main-container" style={{ gap: 10 }}>
        <h1>Analysis History</h1>
        <input
          type="text"
          placeholder="Search by patient email"
          value={patientEmail}
          onChange={handleSearchChange}
          style={{
            width: "30%",
            backgroundColor: "darkgray",
            marginBottom: "3%",
          }}
        />
        {filteredPatients.length > 0 ? (
          filteredPatients.map((analysis, index) => (
            <HistoryItem
              key={index}
              patientName={analysis.patient_email}
              date={analysis.date}
              dominantEmotion={analysis.dominant_emotion}
              analysisSummary={analysis.analysis_summary}
              analysisMode={analysis.analysis_mode}
              sessionDuration={analysis.session_duration}
            />
          ))
        ) : err ? (
          <>
            <span style={{ fontSize: "1.5rem" }} className="err-msg">
              {err}
            </span>
            <img
              src={NoData}
              alt="No Data"
              style={{ width: "10%", height: "10%" }}
            />
          </>
        ) : (
          <span>No matching records found.</span>
        )}
      </div>
    </>
  );
}
