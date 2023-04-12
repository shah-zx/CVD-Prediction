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
  const trainProcess = spawn("python", [
    "train.py",
    patientId,
    systolicBp,
    diastolicBp,
    totalCholesterol,
    triglycerides,
  ]);

  // Handle the output of the train.py process
  trainProcess.stdout.on("data", (data) => {
    console.log(`stdout: ${data}`);
  });

  trainProcess.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  trainProcess.on("close", (trainCode) => {
    console.log(`train.py exited with code ${trainCode}`);

    // Spawn a new Python process to run the accuracyCheck.py script
    const accuracyProcess = spawn("python", ["accuracyCheck.py"]);

    // Handle the output of the accuracyCheck.py process
    accuracyProcess.stdout.on("data", (data) => {
      console.log(`accuracyCheck.py stdout: ${data}`);
    });

    accuracyProcess.stderr.on("data", (data) => {
      console.error(`accuracyCheck.py stderr: ${data}`);
    });

    accuracyProcess.on("close", (accuracyCode) => {
      console.log(`accuracyCheck.py exited with code ${accuracyCode}`);
      // Send a response to the client
      res.send("Form submitted successfully!");
    });
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
