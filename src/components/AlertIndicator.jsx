import React from "react";

export default function AlertIndicator({ status }) {
  return (
    <div className={`alert-circle ${status}`}>
      {status.toUpperCase()}
    </div>
  );
}
