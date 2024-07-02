# Intelligent Financial Agent

This project is an intelligent financial agent that answers questions based on 10-K and 10-Q reports of public companies. It leverages state-of-the-art language models and vector databases to provide accurate and contextually grounded answers.

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/dhruv1702/FinancialAgent.git
    cd FinancialAgent
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    - `SUPABASE_URL`: Your Supabase project URL.
    - `SUPABASE_KEY`: Your Supabase project API key.

    You can set these variables in your shell or create a `.env` file in the root directory of your project:
    ```env
    SUPABASE_URL=your_supabase_url
    SUPABASE_KEY=your_supabase_key
    ```

4. **Download and install Ollama**:
    - Visit [Ollama's website](https://ollama.com) to download and install their CLI tool.
    - Follow the instructions on their website to set up your API key and configure your environment.

## Data Preparation

1. **Download SEC filings**:
    - This project uses `edgartools` to download 10-K and 10-Q reports of the top 50 public companies.
    - Ensure you have your SEC identity set up. You can do this by running:
        ```python
        from edgar import set_identity
        set_identity("Your Name your.email@example.com")
        ```

2. **Chunk and store the data**:
    - The script will chunk the documents into smaller parts and store their embeddings in Supabase for efficient retrieval.

## Running the Application

To run the application, execute the following command:

```bash
python -m src.main
