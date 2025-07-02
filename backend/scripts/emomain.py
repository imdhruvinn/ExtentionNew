from transformers import pipeline
import spacy
from textblob import TextBlob
import nltk
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import re

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download issue: {e}")

# Initialize models
emotion_model = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=None)
sentiment_model = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
nlp = spacy.load("en_core_web_sm")

def scrape_content(url):
    """Scrape content from any webpage."""
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get the title
        title = soup.title.string if soup.title else ""
        
        # Get meta description
        meta_desc = ""
        meta_description = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta_description:
            meta_desc = meta_description.get('content', '')
            
        # Get main content
        content = []
        
        # Get article content if it exists
        article = soup.find('article') or soup.find('main') or soup.find('div', class_=['content', 'article', 'post'])
        if article:
            content.append(article.get_text(separator=' ', strip=True))
        
        # Get all paragraphs
        paragraphs = soup.find_all('p')
        content.extend([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
        
        # Get all headers
        headers = soup.find_all(['h1', 'h2', 'h3'])
        content.extend([h.get_text(strip=True) for h in headers])
        
        # Combine all content
        full_content = ' '.join([title, meta_desc] + content)
        
        return {
            'title': title,
            'meta_description': meta_desc,
            'content': full_content
        }
        
    except Exception as e:
        print(f"Error scraping content: {e}")
        return None

def simple_sentence_split(text):
    """Simple sentence splitter as fallback."""
    # Split on common sentence endings
    sentences = []
    current = ""
    
    for word in text.split():
        current += word + " "
        if word.endswith(('.', '!', '?', '..."', '…')):
            sentences.append(current.strip())
            current = ""
    
    if current:  # Add any remaining text
        sentences.append(current.strip())
    
    return sentences or [text]

def get_combined_emotion_scores(text):
    """Get emotion scores using multiple models and combine them."""
    if not text:
        return None

    # Initialize emotion categories
    emotions = {
        'joy': 0.0,
        'sadness': 0.0,
        'anger': 0.0,
        'fear': 0.0,
        'surprise': 0.0,
        'neutral': 0.0
    }

    try:
        text_lower = text.lower()
        
        # First, detect if this is a humorous context
        humor_indicators = {
            'explicit': [
                r"(funny|hilarious|amusing|lol|haha|hehe)",
                r"(joke|pun|humor)",
                r"(laughing|laughed|laugh)",
                r"\😂|\😹|\🤣|\😆",
            ],
            'casual': [
                r"(gonna|gotta|wanna)",
                r"(like,|i mean,|you know,)",
                r"(honestly|basically|literally)",
            ],
            'exaggeration': [
                r"(absolutely|completely|totally) (ridiculous|absurd)",
                r"(never|ever) in my life",
                r"(worst|best) thing ever",
            ],
            'narrative': [
                r"(you won't believe|guess what)",
                r"(plot twist|turns out)",
                r"(apparently|supposedly)",
            ]
        }

        # Calculate humor score
        humor_score = 0
        for category, patterns in humor_indicators.items():
            weight = 1.5 if category == 'explicit' else 1.0
            matches = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
            humor_score += matches * weight

        is_humorous = humor_score > 0

        # If humorous, immediately reduce sadness baseline
        if is_humorous:
            emotions['sadness'] = -0.5  # Start with negative sadness for humor
            emotions['joy'] += humor_score * 0.4

        # 1. Context Detection
        context_indicators = {
            'sadness': [
                r"hey.*it's me again",
                r"missing you",
                r"since you.*gone",
                r"wish you were here",
                r"not the same without",
                r"everything.*changed",
                r"checking in",
                r"just wanted to tell you"
            ],
            'joy': [
                r"(funny|hilarious|amusing)",
                r"(lol|haha|hehe)",
                r"(happy|exciting) news",
                r"can't wait to",
                r"looking forward to"
            ],
            'anger': [
                r"(angry|mad|furious) about",
                r"(hate|sick of) this",
                r"how dare",
                r"(unfair|unjust)"
            ],
            'fear': [
                r"(scared|afraid) of",
                r"(worried|anxious) about",
                r"(fear|panic) that",
                r"what if.*happens"
            ],
            'surprise': [
                r"(wow|omg|oh my)",
                r"can't believe",
                r"(unexpected|suddenly)",
                r"never thought"
            ]
        }

        # Check dominant context
        context_scores = {emotion: sum(1 for pattern in patterns 
                                     if re.search(pattern, text_lower))
                         for emotion, patterns in context_indicators.items()}
        
        # 2. Deep Narrative Analysis (40% weight)
        narrative_patterns = {
            'joy': [
                # Happiness and celebration
                (r"(happy|glad|excited) (to|that)", 0.8),
                (r"can't wait to", 0.7),
                (r"(love|loving) this", 0.8),
                (r"(wonderful|amazing|fantastic)", 0.7),
                (r"(celebrate|celebrating)", 0.8),
                (r"(proud|proudly)", 0.7),
                (r"(blessed|grateful)", 0.7),
                (r"(fun|funny|hilarious)", 0.7)
            ],
            'sadness': [
                # Loss and grief
                (r"miss(ing)? you", 0.9),
                (r"since you.*gone", 0.9),
                (r"without you", 0.8),
                (r"wish you were", 0.8),
                (r"(hurts|hurting)", 0.7),
                (r"not the same", 0.7),
                (r"everything (changed|different)", 0.7),
                (r"(lonely|alone)", 0.8)
            ],
            'anger': [
                # Rage and frustration
                (r"(angry|mad|furious)", 0.9),
                (r"(hate|hatred)", 0.8),
                (r"(frustrated|annoyed)", 0.7),
                (r"(sick of|fed up)", 0.7),
                (r"(unfair|unjust)", 0.7),
                (r"how dare", 0.8),
                (r"(rage|outrage)", 0.9),
                (r"(blame|fault)", 0.6)
            ],
            'fear': [
                # Fear and anxiety
                (r"(scared|afraid|terrified)", 0.9),
                (r"(fear|fears|feared)", 0.8),
                (r"(worried|anxious)", 0.7),
                (r"(panic|panicking)", 0.8),
                (r"(nightmare|horror)", 0.8),
                (r"(dangerous|threatening)", 0.7),
                (r"(nervous|uneasy)", 0.7),
                (r"what if", 0.6)
            ],
            'surprise': [
                # Shock and amazement
                (r"(surprised|shocked|amazed)", 0.8),
                (r"(unexpected|suddenly)", 0.7),
                (r"(wow|whoa|oh my)", 0.6),
                (r"can't believe", 0.7),
                (r"(unbelievable|incredible)", 0.7),
                (r"(astonished|astonishing)", 0.8),
                (r"(stunned|stunning)", 0.7),
                (r"(plot twist|turn of events)", 0.6)
            ]
        }

        # Process narrative patterns
        for emotion, patterns in narrative_patterns.items():
            for pattern, weight in patterns:
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    emotions[emotion] += weight * min(matches, 3) * 0.4

        # 3. Memory and Context Analysis (20% weight)
        memory_patterns = [
            r"remember when",
            r"used to",
            r"back then",
            r"that time",
            r"those days",
            r"we would always",
            r"you always",
            r"remember how"
        ]

        # Process memories based on context
        dominant_context = max(context_scores.items(), key=lambda x: x[1])[0]
        for pattern in memory_patterns:
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                emotions[dominant_context] += matches * 0.2 * 0.2

        # Modified sentence analysis for better humor handling
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.lower().strip()
            
            # Check for humor in sentence
            has_humor = any(re.search(pattern, sentence) 
                          for patterns in humor_indicators.values() 
                          for pattern in patterns)
            
            if has_humor:
                emotions['joy'] += 0.3
                continue  # Skip regular emotion analysis for humorous sentences
            
            # RoBERTa analysis with humor context
            # We add `truncation=True` to automatically handle text longer than the model's limit.
            sent_emotions = emotion_model(sentence, truncation=True)
            if isinstance(sent_emotions, list) and len(sent_emotions) > 0:
                if isinstance(sent_emotions[0], list):
                    sent_emotions = sent_emotions[0]
                for emotion in sent_emotions:
                    if isinstance(emotion, dict) and 'label' in emotion and 'score' in emotion:
                        label = emotion['label']
                        score = emotion['score']
                        if label in emotions:
                            if is_humorous and label == 'sadness':
                                score *= 0.2  # Drastically reduce sadness in humor
                            elif is_humorous and label == 'joy':
                                score *= 1.5  # Boost joy in humor
                            emotions[label] += score * 0.25

        # Enhanced final adjustments for humor
        if is_humorous:
            emotions['joy'] *= 1.5
            emotions['sadness'] = max(0, emotions['sadness'] * 0.2)  # Stronger sadness reduction
            emotions['neutral'] *= 0.5
            
            # Additional joy boost based on humor intensity
            if humor_score > 2:
                emotions['joy'] *= 1.3
                emotions['sadness'] = max(0, emotions['sadness'] * 0.1)  # Even stronger sadness reduction

        # Normalize scores
        # First, ensure no negative values
        emotions = {k: max(0, v) for k, v in emotions.items()}
        
        # Then normalize to percentages
        total = sum(v for v in emotions.values() if v > 0)
        if total > 0:
            emotions = {k: round((max(v, 0) / total) * 100, 2) for k, v in emotions.items()}

        return emotions

    except Exception as e:
        print(f"Error in emotion analysis: {str(e)}")
        return None

def analyze_content(text):
    """Analyze emotions in text and return a dictionary of scores."""
    if not text:
        print("Error: Empty text. Cannot analyze.")
        return None

    print(f"\nAnalyzing text: {text[:100]}..." if len(text) > 100 else f"\nAnalyzing text: {text}")

    # Split text into sentences
    try:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    except Exception:
        sentences = [text]  # Fallback to analyzing the whole text

    overall_emotions = {
        'joy': 0.0, 'sadness': 0.0, 'anger': 0.0,
        'fear': 0.0, 'surprise': 0.0, 'love': 0.0, 'neutral': 0.0
    }

    valid_analyses = 0
    for sentence in sentences:
        if len(sentence.split()) > 3:  # Only analyze sentences with more than 3 words
            emotions = get_combined_emotion_scores(sentence)
            if emotions:
                valid_analyses += 1
                for emotion, score in emotions.items():
                    if emotion in overall_emotions:
                        overall_emotions[emotion] += score

    # Calculate averages
    if valid_analyses > 0:
        overall_emotions = {k: round(v / valid_analyses, 2) for k, v in overall_emotions.items()}

    # The function now returns the calculated emotion data for server use.
    return overall_emotions

def process_content(url):
    """Process webpage content and analyze emotions."""
    print(f"\nAnalyzing content from: {url}")
    print("\n" + "="*50)
    
    # Scrape the content
    content_data = scrape_content(url)
    
    if not content_data:
        print("Error: Could not fetch content from URL")
        return
    
    # First, show the scraped content
    print("\n📝 SCRAPED CONTENT:")
    print("=" * 50)
    if content_data['title']:
        print(f"\nTitle: {content_data['title']}")
    if content_data['meta_description']:
        print(f"\nDescription: {content_data['meta_description']}")
    
    # Show the main content with a character limit
    content_preview = content_data['content'][:1000] + "..." if len(content_data['content']) > 1000 else content_data['content']
    print(f"\nMain Content:\n{content_preview}")
    
    # Add a separator before emotion analysis
    print("\n" + "="*50)
    print("\nPress Enter to see emotion analysis...")
    input()
    
    # Then show the emotion analysis
    print("\n🔍 EMOTION ANALYSIS")
    print("=" * 50)
    analyze_content(content_data['content'])

def main():
    """Main function to run emotion analysis interactively."""
    while True:
        url = input("\nEnter URL to analyze (or 'quit' to exit): ")
        if url.lower() == 'quit':
            break
        
        process_content(url)
        
        print("\n" + "="*50)
        print("\nAnalysis complete! Press Enter to continue...")
        input()
        print("\n" * 2)

if __name__ == "__main__":
    main()
