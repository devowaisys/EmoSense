import Button from "./Button";
import Icon from "./Icon";
import cancel from "../assets/icons/cancel.png";
import { useContext, useState } from "react";
import { UserContext } from "../UserStore";
import { useNavigate } from "react-router-dom";
import axios from "axios";

export default function LoginPopup({ togglePopup, toggleResetPopup }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [msg, setMsg] = useState("");
  const { addUser, setToken } = useContext(UserContext);
  const navigate = useNavigate();

  async function handleSubmit(evt) {
    evt.preventDefault();
    setError(""); // Clear previous errors
    setMsg(""); // Clear previous success messages
    if (!email.includes("@") || !email.endsWith(".com")) {
      setError("Invalid email");
      return;
    }
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/login_therapist",
        { email, password },
        {
          headers: {
            "Content-Type": "application/json",
          },
          withCredentials: true,
          timeout: 10000, // Timeout after 10 seconds
        }
      );

      const dataJson = response.data;

      if (dataJson.success === true) {
        setError("");
        setMsg(dataJson.message);

        const userData = {
          id: dataJson.therapist.therapist_id,
          full_name: dataJson.therapist.full_name,
          email: dataJson.therapist.email,
        };

        addUser(userData);
        setToken(dataJson.access_token);
        navigate("/home");
      } else {
        setMsg("");
        setError(dataJson.message);
      }
    } catch (err) {
      if (err.response) {
        // Server responded with a status other than 2xx
        setError(
          err.response.data?.message || "An error occurred on the server."
        );
      } else if (err.request) {
        // Request was made but no response received
        setError("Server did not respond. Please try again.");
      } else if (err.code === "ECONNABORTED") {
        // Timeout error
        setError("Request timed out. Please try again.");
      } else {
        // Other errors
        setError(err.message);
      }
      setMsg(""); // Clear success messages
    }
  }

  return (
    <div className="popup-outer-container">
      <Icon
        path={cancel}
        customStyle={{ position: "fixed", top: 30 }}
        onClick={toggleResetPopup}
      />
      <h1>Login</h1>
      {msg ? (
        <span className="success-message">{msg}</span>
      ) : error ? (
        <span className="err-msg">{error}</span>
      ) : null}
      <form className="popup-inner-form">
        <h3>Email</h3>
        <input
          type="email"
          id="email"
          value={email}
          onChange={(evt) => setEmail(evt.target.value)}
          required
        />
        <h3>Password</h3>
        <input
          type="password"
          id="password"
          value={password}
          onChange={(evt) => setPassword(evt.target.value)}
          required
        />
        <div className="button-container">
          <Button
            width="13rem"
            text="Register"
            type="button"
            onClick={togglePopup}
          />
          <Button width="13rem" text="Login" onClick={handleSubmit} />
        </div>
      </form>
    </div>
  );
}
