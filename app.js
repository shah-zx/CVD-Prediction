const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const cors = require("cors");
const app = express();
const port = process.env.PORT || 3000;
app.use(cors());
// Parse incoming requests with JSON payloads
app.use(bodyParser.json());

// Serve the static HTML, CSS and JS files from the public folder
app.use(express.static("public"));

// Handle form submissions
app.post("/submit-form", (req, res) => {
  // Get the form data from the request body
  const {
    patientId,
    systolicBp,
    diastolicBp,
    totalCholesterol,
    triglycerides,
  } = req.body;

  // Spawn a new Python process and pass the form data as arguments
  const pythonProcess = spawn("python", [
    "train.py",
    patientId,
    systolicBp,
    diastolicBp,
    totalCholesterol,
    triglycerides,
  ]);

  // Handle the output of the Python process
  pythonProcess.stdout.on("data", (data) => {
    console.log(`stdout: ${data}`);
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on("close", (code) => {
    console.log(`child process exited with code ${code}`);
  });

  // Send a response to the client
  res.send("Form submitted successfully!");
});

// Start the server
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
