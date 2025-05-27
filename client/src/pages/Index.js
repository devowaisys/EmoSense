import { useEffect, useContext } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../components/Header";
import therapySessionImage from "../assets/icons/therapy-session.png";
import { UserContext } from "../UserStore";

export default function Index() {
  const { user } = useContext(UserContext);
  const navigate = useNavigate();

  useEffect(() => {
    if (user) {
      navigate("/home");
    }
  }, [user, navigate]); // Dependency array ensures this runs when `user` changes

  return (
    <>
      <Header />
      <div className="main-container">
        <main>
          <div className="left-container">
            <img
              src={therapySessionImage}
              alt="Therapy Session"
              style={{ width: "50%", height: "100%" }}
            />
          </div>
          <div className="right-container">
            <h2>Welcome to Emo Sense!</h2>
            <h3 style={{ textAlign: "center" }}>
              Unlock Emotional Insights, Empower Your Therapy Sessions, and
              Transform Client Understanding in Real-Time.
            </h3>
          </div>
        </main>
      </div>
    </>
  );
}
