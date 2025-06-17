# Streamlit App for LinkedIn Profile Scraper and Matcher (No Face Matching)
# Requirements: streamlit, selenium, beautifulsoup4, sentence-transformers, fuzzywuzzy, requests

import os
import json
import time
import requests
import numpy as np
import streamlit as st
from urllib.parse import urlparse, parse_qs, quote_plus
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure Streamlit
st.set_page_config(page_title="LinkedIn Profile Matcher", layout="wide")
st.title("🔎 LinkedIn Profile Matcher (No Face Matching)")

# Set up Selenium browser
@st.cache_resource
def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/114.0.0.0 Safari/537.36")
    return webdriver.Chrome(options=options)

driver = get_driver()

def google_drive_to_direct_url(view_url):
    if 'drive.google.com' in view_url:
        file_id = view_url.split('/d/')[1].split('/')[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return view_url

def clean_url(possibly_wrapped_url):
    if 'google.com/url?q=' in possibly_wrapped_url:
        return parse_qs(urlparse(possibly_wrapped_url).query).get('q', [possibly_wrapped_url])[0]
    return possibly_wrapped_url

def search_linkedin(name, intro, location):
    query = f'{name} {intro if intro else ""} {location if location else ""} site:linkedin.com/in/'
    driver.get("https://html.duckduckgo.com/html/")
    time.sleep(1)

    # Fill and submit the search form
    search_input = driver.find_element(By.NAME, "q")
    search_input.clear()
    search_input.send_keys(query)
    search_input.submit()
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
        
    links = [a['href'] for a in soup.find_all('a', href=True) if 'linkedin.com/in/' in a['href']]
    return list(set(links))[:3]

def scrape_profile(url):
    driver.get(url)
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    name = soup.find('h1')
    headline = soup.find('div', class_='text-body-medium')
    location = soup.find('span', class_='text-body-small')
    return {
        'url': url,
        'name': name.get_text(strip=True) if name else '',
        'bio': headline.get_text(strip=True) if headline else '',
        'location': location.get_text(strip=True) if location else ''
    }

def score_profiles(input_data, scraped_profiles):
    scores = []
    for profile in scraped_profiles:
        name_score = fuzz.ratio(input_data['name'], profile['name']) / 100
        bio_score = util.cos_sim(
            model.encode(input_data['intro'] if input_data['intro'] else ""),
            model.encode(profile['bio'])
        ).item() if profile['bio'] else 0
        location_score = fuzz.ratio(input_data.get('timezone', ''), profile['location']) / 100 if profile['location'] else 0

        total_score = 0.5 * name_score + 0.3 * bio_score + 0.2 * location_score
        scores.append({"profile": profile, "score": total_score})

    scores.sort(key=lambda x: x['score'], reverse=True)
    return scores[:5]

def run_matching_pipeline(personas):
    results = []
    for person in personas:
        st.write(f"🔍 Searching for: **{person['name']}**")
        urls = search_linkedin(person['name'], person.get('intro', ''), person.get('timezone', ''))
        if not urls:
            st.warning(f"❌ No LinkedIn profiles found for: {person['name']}")
            results.append({"input_persona": person['name'], "matches": []})
            continue
        profiles = [scrape_profile(url) for url in urls]
        top_matches = score_profiles(person, profiles)
        results.append({"input_persona": person['name'], "matches": top_matches})
    return results

# UI
uploaded_file = st.file_uploader("📤 Upload JSON file with persona data", type="json")
if uploaded_file is not None:
    personas = json.load(uploaded_file)
    with st.spinner("Running profile matching..."):
        results = run_matching_pipeline(personas)
    st.success("✅ Matching complete!")
    for res in results:
        st.markdown(f"### 🎯 Results for: {res['input_persona']}")
        if not res['matches']:
            st.info("No matches found.")
        for match in res['matches']:
            profile = match['profile']
            clean_link = clean_url(profile['url'])
            st.markdown(f"- [{profile['name']}]({clean_link}) — Score: **{match['score']:.2f}**")
else:
    st.info("Upload a `.json` file to begin.")
