import Button from "./Button";
import Icon from "./Icon";
import cancel from "../assets/icons/cancel.png";
import { useState } from "react";
import axios from "axios";

export default function RegisterPopup({ togglePopup, toggleResetPopup }) {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState("");
  const [error, setError] = useState("");
  const [passwordError, setPasswordError] = useState("");

  // Password validation function
  const validatePassword = (pass) => {
    if (pass.length < 8) {
      return "Password must be at least 8 characters";
    }
    if (!/[A-Z]/.test(pass)) {
      return "Password must contain at least one uppercase letter";
    }
    if (!/[!@#$%^&*(),.?":{}|<>]/.test(pass)) {
      return "Password must contain at least one special character";
    }
    return "";
  };

  const handlePasswordChange = (evt) => {
    const value = evt.target.value;
    setPassword(value);

    // Validate as user types
    const errorMsg = validatePassword(value);
    setPasswordError(errorMsg);

    // Clear general error when typing
    if (error && value) {
      setError("");
    }
  };

  async function handleSubmit(evt) {
    evt.preventDefault();
    setMsg(""); // Clear previous success messages
    setError(""); // Clear previous error messages

    // Validate email
    if (!email.includes("@") || !email.includes(".")) {
      setError("Invalid email");
      return;
    }

    // Validate password
    const passwordValidation = validatePassword(password);
    if (passwordValidation) {
      setPasswordError(passwordValidation);
      return;
    }

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/api/add_therapist",
        {
          full_name: fullName,
          email: email,
          password: password,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
          timeout: 10000,
        }
      );

      const dataJson = response.data;

      if (dataJson.success === true) {
        setError("");
        setPasswordError("");
        setMsg(dataJson.message);
      } else {
        setMsg("");
        setError(dataJson.message);
      }
    } catch (err) {
      if (err.response) {
        setError(
          err.response.data?.message || "An error occurred on the server."
        );
      } else if (err.request) {
        setError("Server did not respond. Please try again.");
      } else if (err.code === "ECONNABORTED") {
        setError("Request timed out. Please try again.");
      } else {
        setError(err.message);
      }
      setMsg("");
    }
  }

  return (
    <div className={"popup-outer-container"}>
      <Icon
        path={cancel}
        customStyle={{ position: "fixed", top: 30 }}
        onClick={toggleResetPopup}
      />
      <h1>Register</h1>
      {msg ? (
        <span className={"success-message"}>{msg}</span>
      ) : error ? (
        <span className={"err-msg"}>{error}</span>
      ) : null}
      <form className={"popup-inner-form"}>
        <h3>Full Name</h3>
        <input
          type={"text"}
          id={"name"}
          onChange={(evt) => setFullName(evt.target.value)}
          value={fullName}
          required
        />

        <h3>Email</h3>
        <input
          type={"email"}
          id={"email"}
          onChange={(evt) => setEmail(evt.target.value)}
          value={email}
          required
        />

        <h3>Password</h3>
        <input
          type={"password"}
          id={"password"}
          onChange={handlePasswordChange}
          value={password}
          required
        />
        {passwordError && (
          <div className="password-requirements">
            <span className={"err-msg"}>{passwordError}</span>
            <ul className="requirement-list">
              <li className={password.length >= 8 ? "valid" : "invalid"}>
                Minimum 8 characters
              </li>
              <li className={/[A-Z]/.test(password) ? "valid" : "invalid"}>
                At least 1 uppercase letter
              </li>
              <li
                className={
                  /[!@#$%^&*(),.?":{}|<>]/.test(password) ? "valid" : "invalid"
                }
              >
                At least 1 special character
              </li>
            </ul>
          </div>
        )}

        <div className={"button-container"}>
          <Button width={"13rem"} text={"Login"} onClick={togglePopup} />
          <Button width={"13rem"} text={"Register"} onClick={handleSubmit} />
        </div>
      </form>
    </div>
  );
}
