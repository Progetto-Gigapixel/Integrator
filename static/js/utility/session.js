export function recoverSessionVariables(elementId = "flask-session-data") {
  const sessionElement = document.getElementById(elementId);
  if (!sessionElement) {
    console.warn(`Element #${elementId} not found!`);
    return {};
  }

  const sessionData = {};
  const attributes = sessionElement.dataset; // Get all data-* attributes

  // Convert all data-* attributes to an object
  for (const key in attributes) {
    // Convert keys from camelCase to snake_case (optional)
    const sessionKey = key.replace(/([A-Z])/g, "_$1").toLowerCase();
    try {
      // Parse JSON if the value is a JSON string (e.g., numbers, booleans, arrays)
      sessionData[sessionKey] = JSON.parse(attributes[key]);
    } catch (e) {
      // Fallback to raw string if not JSON
      sessionData[sessionKey] = attributes[key];
    }
  }

  return sessionData;
}
