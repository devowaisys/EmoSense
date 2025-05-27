import profileUserIcon from "../assets/icons/profile-user.png";
import Icon from "./Icon";
import Button from "./Button";
import { useContext } from "react";
import { UserContext } from "../UserStore";

export default function NavigationMenu({ onGetStarted, onProfile }) {
  const { user } = useContext(UserContext);

  return (
    <nav>
      <ul>
        <li>
          {user ? (
            <a href={"/history"}>History</a>
          ) : (
            <Button text={"Get Started"} onClick={onGetStarted} />
          )}
        </li>
        <li>
          <a href={"/about"}>About</a>
        </li>
        <li>
          <a href={"/privacy"}>Privacy Policy</a>
        </li>
        <li>
          <Icon
            path={profileUserIcon}
            onClick={user ? onProfile : onGetStarted}
          />
        </li>
      </ul>
    </nav>
  );
}
