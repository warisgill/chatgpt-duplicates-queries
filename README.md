# ChatGPT Duplicate Queries Detector

This repository contains a tool for detecting duplicate queries in ChatGPT conversations. Follow the steps below to set up your environment, download your ChatGPT chat data, and find duplicates.


## Privacy Note

🛡 **Your Data Privacy**: Please note that your original chat data or queries are **not shared** when you send the results. The files you share contain only statistical information about duplicates, ensuring your data's privacy and confidentiality.


## Installation

1. **Create and Activate a Python Environment**  
   Begin by creating a new Python environment named `llm11` and then activate it. These steps ensure that your project dependencies do not interfere with your global Python setup.

   ```bash
   conda create --name llm11 python=3
   conda activate llm11
   ```

2. **Install Required Libraries**  
   With your environment activated, install the necessary Python libraries using pip.

   ```bash
   pip install pandas torch sentence-transformers tqdm diskcache fire einops
   ```

## Preparing Your Chat Data

3. **Download ChatGPT Conversations**  
   To download your ChatGPT chat data, watch the instructional video at [https://www.youtube.com/watch?v=SRg3sWyOhzc](https://www.youtube.com/watch?v=SRg3sWyOhzc). This video will guide you through the process of exporting your conversations from ChatGPT.

4. **Extract and Prepare the Data**  
   Once you have downloaded your chat data, unzip the file and move `conversations.json` into the `private_csv_json` directory within this project.

## Detecting Duplicates

5. **Run the Detection Script**  
   Execute the main script to start the process of finding duplicate queries in your chat data.

   ```bash
   python main.py
   ```

6. **Inspect the Results**  
   After running the script, check the `only_duplicates.csv` file in the `private_csv_json` folder to review the duplicate queries detected.

## Sharing Results

7. **Send Results**  
   Please send the `final_results.csv` and `trace_duplicates.csv` files located in the `result_csv` folder to `waris@vt.edu`.

   If possible, also include two interesting examples of duplicate queries from the `only_duplicates.csv` file in your email.

---





