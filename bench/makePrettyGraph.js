const fs = require('fs');
const QuickChart = require('quickchart-js');
const path = require('path');

// Load benchmark results
const resultsFile = path.join(__dirname, 'benchmark_results.json');
let results = {};

// Load results from JSON file
if (fs.existsSync(resultsFile)) {
    results = JSON.parse(fs.readFileSync(resultsFile, 'utf8'));
} else {
    console.error('benchmark_results.json not found.');
    process.exit(1);
}

// Prepare chart data
let data = Object.keys(results).map(pattern => ({
    statesLength: results[pattern]['states_length'],
    averageTime: results[pattern]['average_time'],
    outlinesTime: results[pattern]['outlines_time'] || 0  // Ensure outlines_time is available
}));

// Sort data by states length
data.sort((a, b) => a.statesLength - b.statesLength);

// Extract sorted state lengths, average times, and outlines times
const stateLengths = data.map(item => item.statesLength);  // X-axis (state lengths)
const times = data.map(item => item.averageTime);          // Y-axis (average times for your implementation)
const outlinesTimes = data.map(item => item.outlinesTime); // Y-axis (Outlines library times)

// Create the dataset for the graph, including the baseline (0.02 seconds)
const datasets = [
    {
        label: 'Faster-Outlines (num-threads: auto)',
        data: times,
        fill: false,
        borderColor: 'rgb(255, 99, 132)',  // Red color for your implementation
        tension: 0.1
    },
    {
        label: 'Outlines Library',
        data: outlinesTimes,
        fill: false,
        borderColor: 'rgb(54, 162, 235)',  // Blue color for Outlines library
        tension: 0.1
    },
    {
        label: 'Time Inference Is Blocked (avg)',
        data: Array(stateLengths.length).fill(0.02),  // Flat line at 0.02 seconds ( Average )
        fill: false,
        borderColor: 'rgb(255, 206, 86)',  // Yellow color for the baseline
        borderDash: [5, 5],  // Dashed line
        tension: 0.1
    }
];

// Create the chart with dark mode styling
const chart = new QuickChart();
chart.setConfig({
    type: 'line',
    data: {
        labels: stateLengths,  // X-axis will be state lengths
        datasets: datasets     // Y-axis will be average times for both your implementation and Outlines library
    },
    options: {
        responsive: true,
        plugins: {
            title: {
                display: true,
                text: 'Regex Compilation Time vs. State Length (Single Thread)',
                color: '#ffffff'  // Title color for dark mode
            },
            legend: {
                labels: {
                    color: '#ffffff'  // Legend text color for dark mode
                }
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Regex State Length',
                    color: '#ffffff'  // X-axis label color
                },
                ticks: {
                    color: '#ffffff'  // X-axis ticks color
                },
                type: 'logarithmic'
            },
            y: {
                title: {
                    display: true,
                    text: 'Average Compilation Time (s)',
                    color: '#ffffff'  // Y-axis label color
                },
                ticks: {
                    color: '#ffffff'  // Y-axis ticks color
                },
                type: 'logarithmic'
            }
        }
    }
}).setBackgroundColor('#2c2c2c')  // Dark mode background
  .setWidth(800)
  .setHeight(600);

// Render and save chart as an image
chart.toFile('assets/benchmark.png')
    .then(() => {
        console.log('Graph generated and saved as assets/benchmark.png');
    })
    .catch((error) => {
        console.error('Error generating graph:', error);
    });

// Helper function to generate random color for the dataset (used for dataset styling)
function getRandomColor() {
    const r = Math.floor(Math.random() * 255);
    const g = Math.floor(Math.random() * 255);
    const b = Math.floor(Math.random() * 255);
    return `rgb(${r},${g},${b})`;
}
