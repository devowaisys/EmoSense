import { useContext } from "react";
import { UserContext } from "../UserStore";

export default function Logo() {
  const { user } = useContext(UserContext);

  return (
    <a className="logo" href={user ? "/home" : "/"}>
      &nbsp;
    </a>
  );
}
