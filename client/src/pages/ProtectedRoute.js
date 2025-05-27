import { useContext } from "react";
import { UserContext } from "../UserStore";
import { Navigate } from "react-router-dom";

export default function ProtectedRoute({ children }) {
  const { user } = useContext(UserContext);

  if (!user || !user.id) {
    return <Navigate to="/" replace />;
  }

  return children;
}
