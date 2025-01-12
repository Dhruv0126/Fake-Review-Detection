document.getElementById('signup-btn').addEventListener('click', function(event) {
  event.preventDefault();
  // Show Main Detection Page
  document.getElementById('signup-page').style.display = 'none';
  document.getElementById('main-page').style.display = 'block';
});

document.getElementById('analyze-btn').addEventListener('click', function() {
  const reviewInput = document.getElementById('review-input').value;
  if (reviewInput.trim() === '') {
    alert('Please enter a review to analyze.');
    return;
  }
  analyzeReview(reviewInput);
});

document.getElementById('upload-btn').addEventListener('click', function() {
  uploadCSV();
});

document.getElementById('explain-btn').addEventListener('click', function() {
  explainPrediction();
});

async function analyzeReview(review) {
  try {
    // Show loading spinner
    setLoadingState(true);
    const response = await fetch('/analyze_review', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ review })
    });

    if (response.ok) {
      const data = await response.json();
      // Display analysis result
      document.getElementById('output').innerText = `Percentage of Fake Review: ${data.fake_percentage}% | Percentage of Real Review: ${data.real_percentage}%`;
    } else {
      alert('Failed to analyze the review. Please try again later.');
    }
  } catch (error) {
    console.error('Error analyzing review:', error);
    alert('An error occurred while analyzing the review.');
  } finally {
    // Hide loading spinner
    setLoadingState(false);
  }
}

async function uploadCSV() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.csv';
  input.onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Show loading spinner
    setLoadingState(true);

    const formData = new FormData();
    formData.append('csv_file', file);

    try {
      const response = await fetch('/upload_csv', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        alert(data.message); // Show success message
      } else {
        alert('Failed to upload the CSV file. Please try again.');
      }
    } catch (error) {
      console.error('Error uploading CSV:', error);
      alert('An error occurred while uploading the CSV file.');
    } finally {
      // Hide loading spinner
      setLoadingState(false);
    }
  };
  input.click();
}

async function explainPrediction() {
  try {
    const response = await fetch('/explain_prediction', {
      method: 'GET'
    });

    const data = await response.json();

    if (response.ok) {
      alert(data.explanation);  // Show explanation of the review
    } else {
      alert('Failed to fetch explanation. Please try again.');
    }
  } catch (error) {
    console.error('Error explaining prediction:', error);
    alert('An error occurred while fetching the explanation.');
  }
}

// Utility function to toggle loading state
function setLoadingState(isLoading) {
  const spinner = document.getElementById('loading-spinner');
  const buttons = document.querySelectorAll('button');

  if (isLoading) {
    // Show spinner and disable buttons
    spinner.style.display = 'inline-block';
    buttons.forEach(button => button.disabled = true);
  } else {
    // Hide spinner and enable buttons
    spinner.style.display = 'none';
    buttons.forEach(button => button.disabled = false);
  }
}
