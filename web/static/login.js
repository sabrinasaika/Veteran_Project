"use strict";

const LOGIN_ERRORS = {
  invalid: "Incorrect username or password.",
  locked: "Too many failed attempts. Please wait 10 minutes and try again.",
};

const errorCode = new URLSearchParams(window.location.search).get("error");
if (errorCode && LOGIN_ERRORS[errorCode]) {
  const el = document.getElementById("login-error");
  el.textContent = LOGIN_ERRORS[errorCode];
  el.hidden = false;
}
