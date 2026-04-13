import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Waste Segregation Assistant",
    page_icon="♻️",
    layout="centered"
)

# --------------------------------------------------
# LOAD MODEL AND CLASS NAMES
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/waste_model.keras")

@st.cache_data
def load_class_names():
    with open("model/class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

# --------------------------------------------------
# LANGUAGE TEXT
# --------------------------------------------------
ui_text = {
    "English": {
        "title": "♻️ AI-Based Smart Waste Segregation Assistant",
        "subtitle": "Offline waste classification with bilingual guidance for disposal, health, renewable energy, and smart city impact.",
        "language": "Select Language",
        "source": "Choose Image Source",
        "source_option": "Select input method:",
        "upload": "Upload Image",
        "camera": "Use Camera",
        "upload_label": "Upload an image",
        "camera_label": "Capture a waste image",
        "prediction": "Prediction Result",
        "predicted": "Predicted Waste Type",
        "confidence": "Confidence",
        "low_conf": "Low confidence prediction. Please verify manually with a clearer image.",
        "category": "Waste Category",
        "gov_stream": "Segregation Stream",
        "disposal": "Disposal Method",
        "impact": "Environmental Impact",
        "recycle": "Recycling / Reuse Opportunity",
        "energy": "Composting / Renewable Energy Potential",
        "health": "Health Benefit",
        "smart_city": "Smart City Benefit",
        "society": "Benefit to Society and Daily Life",
        "judge_note": "Why This Project Matters",
        "verify": "Verification Note",
        "info_wait": "Please upload an image or use the camera to begin prediction.",
        "footer": "This version works fully offline. It does not depend on Gemini or any online API."
    },
    "Tamil": {
        "title": "♻️ செயற்கை நுண்ணறிவு அடிப்படையிலான ஸ்மார்ட் கழிவு வகைப்படுத்தும் உதவி அமைப்பு",
        "subtitle": "இணையம் இல்லாமலும் இயங்கும் கழிவு வகைப்படுத்தல் செயலி. அகற்றும் முறை, ஆரோக்கியம், புதுப்பிக்கத்தக்க ஆற்றல், ஸ்மார்ட் நகர நன்மைகள் ஆகியவை தமிழிலும் ஆங்கிலத்திலும் வழங்கப்படும்.",
        "language": "மொழியைத் தேர்ந்தெடுக்கவும்",
        "source": "படத்தின் மூலம் தேர்வு செய்யவும்",
        "source_option": "உள்ளீட்டு முறையைத் தேர்ந்தெடுக்கவும்:",
        "upload": "படத்தை பதிவேற்று",
        "camera": "கேமராவைப் பயன்படுத்து",
        "upload_label": "படத்தை பதிவேற்றவும்",
        "camera_label": "கழிவு பொருளின் படத்தை எடுக்கவும்",
        "prediction": "கணிப்பு முடிவு",
        "predicted": "கணிக்கப்பட்ட கழிவு வகை",
        "confidence": "நம்பகத்தன்மை",
        "low_conf": "குறைந்த நம்பகத்தன்மை கொண்ட கணிப்பு. தெளிவான படத்துடன் கைமுறையாக சரிபார்க்கவும்.",
        "category": "கழிவு தன்மை",
        "gov_stream": "அரசு பிரிப்பு வகை",
        "disposal": "அகற்றும் முறை",
        "impact": "சுற்றுச்சூழல் பாதிப்பு / நன்மை",
        "recycle": "மறுசுழற்சி / மறுபயன்பாட்டு வாய்ப்பு",
        "energy": "கம்போஸ்ட் / புதுப்பிக்கத்தக்க ஆற்றல் வாய்ப்பு",
        "health": "ஆரோக்கிய நன்மை",
        "smart_city": "ஸ்மார்ட் நகர நன்மை",
        "society": "சமூக மற்றும் தனிநபர் நன்மை",
        "judge_note": "இந்த திட்டம் ஏன் முக்கியம்",
        "verify": "சரிபார்ப்பு குறிப்பு",
        "info_wait": "கணிப்பை தொடங்க படம் பதிவேற்றவும் அல்லது கேமராவைப் பயன்படுத்தவும்.",
        "footer": "இந்த பதிப்பு முழுமையாக ஆஃப்லைனில் செயல்படும். Gemini அல்லது வேறு இணைய API தேவையில்லை."
    }
}

# --------------------------------------------------
# WASTE KNOWLEDGE BASE
# --------------------------------------------------
waste_data = {
    "biological": {
        "name_en": "Biological",
        "name_ta": "உயிரியல் கழிவு",
        "category_en": "Biodegradable",
        "category_ta": "உயிர்ச்சிதைவு அடையும் கழிவு",
        "stream_en": "Biodegradable / Wet Waste",
        "stream_ta": "உயிர்ச்சிதைவு அடையும் / ஈரக் கழிவு",
        "disposal_en": "This is organic waste. It should be sent for composting or biomethanation instead of being mixed with dry waste.",
        "disposal_ta": "இது இயற்கை சார்ந்த உயிரியல் கழிவு. இதை உலர் கழிவுடன் கலக்காமல் கம்போஸ்ட் அல்லது பயோமீதனேஷன் முறைக்கு அனுப்ப வேண்டும்.",
        "impact_en": "Proper handling reduces landfill load, bad smell, and methane release from unmanaged dumping.",
        "impact_ta": "சரியான முறையில் கையாளப்பட்டால் குப்பை மேடு சுமை, துர்நாற்றம் மற்றும் கட்டுப்பாடற்ற மீத்தேன் உமிழ்வு குறையும்.",
        "recycle_en": "It can be converted into compost, manure, or slurry for agriculture and gardens.",
        "recycle_ta": "இதை கம்போஸ்ட், உரம் அல்லது வேளாண்மை மற்றும் தோட்ட பயன்பாட்டிற்கான சழை வடிவமாக மாற்றலாம்.",
        "energy_en": "Yes. This waste supports composting and can also be used in biogas or biomethanation systems as a renewable-energy source.",
        "energy_ta": "ஆம். இந்த கழிவு கம்போஸ்டாக மாற்றப்படுவதுடன், பையோகேஸ் அல்லது பயோமீதனேஷன் முறையில் புதுப்பிக்கத்தக்க ஆற்றல் ஆதாரமாகவும் பயன்படும்.",
        "health_en": "Segregating this waste quickly helps reduce flies, insects, foul odour, and disease-spreading conditions.",
        "health_ta": "இந்த கழிவை உடனே பிரித்தால் ஈக்கள், பூச்சிகள், துர்நாற்றம் மற்றும் நோய் பரவக்கூடிய சூழல் குறையும்.",
        "smart_city_en": "Decentralized composting and biogas systems help smart cities reduce transport burden and manage waste near the source.",
        "smart_city_ta": "மையமற்ற கம்போஸ்ட் மற்றும் பையோகேஸ் அமைப்புகள், போக்குவரத்து சுமையைக் குறைத்து, மூலத்திலேயே கழிவை நிர்வகிக்க உதவுவதால் ஸ்மார்ட் நகரத்திற்கு உதவுகின்றன.",
        "society_en": "This supports cleaner streets, healthier neighborhoods, local compost use, and better household waste habits.",
        "society_ta": "இது சுத்தமான தெருக்கள், ஆரோக்கியமான குடியிருப்புகள், உள்ளூர் கம்போஸ்ட் பயன்பாடு மற்றும் நல்ல வீட்டு கழிவு பழக்கங்களை உருவாக்க உதவுகிறது.",
        "judge_note_en": "Organic waste segregation is the base for health-focused systems, renewable-energy projects like biogas, and smart-city waste management.",
        "judge_note_ta": "உயிரியல் கழிவைப் பிரித்தல் என்பது ஆரோக்கியம் சார்ந்த திட்டங்கள், பையோகேஸ் போன்ற புதுப்பிக்கத்தக்க ஆற்றல் திட்டங்கள் மற்றும் ஸ்மார்ட் நகர கழிவு நிர்வாகத்தின் அடித்தளம் ஆகும்.",
        "verify_en": "If food, plastic, and metal are mixed together, manual verification is recommended.",
        "verify_ta": "உணவு, பிளாஸ்டிக், உலோகம் போன்றவை கலந்து இருந்தால் கைமுறையாக சரிபார்க்க பரிந்துரைக்கப்படுகிறது."
    },
    "cardboard": {
        "name_en": "Cardboard",
        "name_ta": "கார்ட்போர்டு",
        "category_en": "Biodegradable but handled as dry recyclable waste",
        "category_ta": "உயிர்ச்சிதைவு அடையும்; ஆனால் உலர் மறுசுழற்சி கழிவாக கையாளப்படுகிறது",
        "stream_en": "Dry Recyclable Waste",
        "stream_ta": "உலர் மறுசுழற்சி கழிவு",
        "disposal_en": "Keep it dry, fold it if needed, and send it for paper/cardboard recycling.",
        "disposal_ta": "இதை உலர வைத்துப் பாதுகாத்து, தேவையெனில் மடித்து, காகித / கார்ட்போர்டு மறுசுழற்சிக்கு அனுப்ப வேண்டும்.",
        "impact_en": "Recycling cardboard reduces dumping and helps conserve raw material used for paper products.",
        "impact_ta": "கார்ட்போர்டை மறுசுழற்சி செய்தால் குப்பை மேடுகள் குறைந்து, காகிதப் பொருட்களுக்கு தேவையான மூலப்பொருள் சேமிக்கப்படும்.",
        "recycle_en": "It can be reused for packaging or recycled into paper boards and cartons.",
        "recycle_ta": "இதை பொதியிட பயன்படுத்தலாம் அல்லது மறுசுழற்சி செய்து காகித பலகை, பெட்டி போன்றவற்றாக மாற்றலாம்.",
        "energy_en": "It is not meant for composting in this app flow. Its best value is through dry recycling.",
        "energy_ta": "இந்த செயலியின் போக்கில் இதை கம்போஸ்டுக்கு பரிந்துரைக்கவில்லை. இதன் சிறந்த பயன் உலர் மறுசுழற்சியில்தான் உள்ளது.",
        "health_en": "Proper segregation keeps surroundings cleaner and avoids mixed waste contamination.",
        "health_ta": "சரியான பிரிப்பு, சுற்றுப்புறத்தை சுத்தமாக வைத்துக் கொண்டு, கலப்பு கழிவு மாசுபாட்டைத் தடுக்கிறது.",
        "smart_city_en": "Source segregation improves material recovery and reduces landfill dependency in urban systems.",
        "smart_city_ta": "மூலத்திலேயே பிரிப்பது, பொருள் மீட்பை அதிகரித்து, நகரங்களில் குப்பை மேடுகளைச் சார்ந்திருப்பதை குறைக்கிறது.",
        "society_en": "This promotes recycling jobs, cleaner storage, and better household waste discipline.",
        "society_ta": "இது மறுசுழற்சி வேலைவாய்ப்பு, சுத்தமான சேமிப்பு மற்றும் வீட்டு கழிவு ஒழுக்கத்தை மேம்படுத்துகிறது.",
        "judge_note_en": "Dry recyclable segregation is essential for circular economy models and smart-city material recovery systems.",
        "judge_note_ta": "உலர் மறுசுழற்சி கழிவைப் பிரிப்பது சுற்றுச்சுழற்சி பொருளாதாரம் மற்றும் ஸ்மார்ட் நகர பொருள் மீட்பு அமைப்புகளுக்கு மிகவும் அவசியமானது.",
        "verify_en": "If it is wet, oily, or mixed with food waste, manual sorting may be required.",
        "verify_ta": "இது நனைந்திருந்தாலோ, எண்ணெய் படிந்திருந்தாலோ அல்லது உணவுக் கழிவுடன் கலந்திருந்தாலோ கைமுறை பிரிப்பு தேவைப்படலாம்."
    },
    "metal": {
        "name_en": "Metal",
        "name_ta": "உலோகம்",
        "category_en": "Non-biodegradable",
        "category_ta": "உயிர்ச்சிதைவு அடையாத கழிவு",
        "stream_en": "Dry Recyclable Waste",
        "stream_ta": "உலர் மறுசுழற்சி கழிவு",
        "disposal_en": "Collect separately and send it to scrap dealers or authorized recycling channels.",
        "disposal_ta": "இதை தனியாக சேகரித்து, குப்பை இரும்பு விற்பனையாளர் அல்லது அங்கீகரிக்கப்பட்ட மறுசுழற்சி மையத்துக்கு அனுப்ப வேண்டும்.",
        "impact_en": "Metal recycling reduces mining pressure, saves energy, and lowers environmental damage.",
        "impact_ta": "உலோகத்தை மறுசுழற்சி செய்தால் சுரங்க அகழ்வு தேவையும், ஆற்றல் பயன்பாடும், சுற்றுச்சூழல் சேதமும் குறையும்.",
        "recycle_en": "Metal can be melted and reused in manufacturing many products.",
        "recycle_ta": "உலோகத்தை உருக்கி பல தயாரிப்புகளுக்காக மீண்டும் பயன்படுத்தலாம்.",
        "energy_en": "This waste is not for composting. Its main sustainability value comes from recycling and material recovery.",
        "energy_ta": "இந்த கழிவு கம்போஸ்டுக்கு பொருந்தாது. இதன் நிலைத்தன்மை மதிப்பு மறுசுழற்சி மற்றும் பொருள் மீட்பில்தான் உள்ளது.",
        "health_en": "Segregation prevents sharp or hazardous items from mixing with household wet waste.",
        "health_ta": "உலோகத்தை பிரிப்பது கூர்மையான அல்லது ஆபத்தான பொருட்கள் வீட்டு ஈரக் கழிவுடன் கலப்பதைத் தடுக்கிறது.",
        "smart_city_en": "Recovering metals supports efficient urban resource management and reduces municipal waste load.",
        "smart_city_ta": "உலோக மீட்பு நகர வள மேலாண்மையை மேம்படுத்தி, நகராட்சி கழிவு சுமையை குறைக்க உதவுகிறது.",
        "society_en": "It supports scrap-value income, safer handling, and a stronger recycling economy.",
        "society_ta": "இது குப்பை மதிப்பு வருமானம், பாதுகாப்பான கையாளுதல், மற்றும் வலுவான மறுசுழற்சி பொருளாதாரத்தை ஆதரிக்கிறது.",
        "judge_note_en": "Metal segregation connects waste management with industry reuse, resource efficiency, and circular economy outcomes.",
        "judge_note_ta": "உலோகக் கழிவு பிரிப்பு, கழிவு மேலாண்மையை தொழில் மறுபயன்பாடு, வளச் சேமிப்பு மற்றும் சுற்றுசுழற்சி பொருளாதாரத்துடன் இணைக்கிறது.",
        "verify_en": "Sharp, rusted, or mixed objects should be handled carefully and verified manually.",
        "verify_ta": "கூர்மையான, சுரண்டிய அல்லது கலந்த பொருட்கள் கவனமாக கையாளப்பட்டு கைமுறையாகச் சரிபார்க்கப்பட வேண்டும்."
    },
    "paper": {
        "name_en": "Paper",
        "name_ta": "காகிதம்",
        "category_en": "Biodegradable but handled as dry recyclable waste",
        "category_ta": "உயிர்ச்சிதைவு அடையும்; ஆனால் உலர் மறுசுழற்சி கழிவாக கையாளப்படுகிறது",
        "stream_en": "Dry Recyclable Waste",
        "stream_ta": "உலர் மறுசுழற்சி கழிவு",
        "disposal_en": "Keep it clean and dry, then send it for paper recycling.",
        "disposal_ta": "இதை சுத்தமாகவும் உலர்ந்தும் வைத்து, காகித மறுசுழற்சி மையத்துக்கு அனுப்ப வேண்டும்.",
        "impact_en": "Paper recycling helps conserve natural resources and reduces mixed waste load.",
        "impact_ta": "காகித மறுசுழற்சி இயற்கை வளங்களை பாதுகாக்க உதவுவதுடன், கலப்பு கழிவு சுமையையும் குறைக்கிறது.",
        "recycle_en": "It can be recycled into notebooks, cartons, boards, and paper products.",
        "recycle_ta": "இதை நோட்டுப் புத்தகம், பெட்டி, பலகை மற்றும் பிற காகிதப் பொருட்களாக மறுசுழற்சி செய்யலாம்.",
        "energy_en": "This is best handled through recycling. Do not mix it with wet waste.",
        "energy_ta": "இதை மறுசுழற்சியே சிறந்த முறையில் கையாளும். இதை ஈரக் கழிவுடன் கலக்க வேண்டாம்.",
        "health_en": "Segregation improves cleanliness and avoids foul contamination when mixed with organic waste.",
        "health_ta": "சரியான பிரிப்பு சுத்தத்தை அதிகரித்து, உயிரியல் கழிவுடன் கலந்தால் ஏற்படும் அழுக்கு நிலையைத் தடுக்கிறது.",
        "smart_city_en": "Paper recovery improves urban recycling efficiency and reduces landfill pressure.",
        "smart_city_ta": "காகித மீட்பு நகர மறுசுழற்சி திறனை உயர்த்தி, குப்பை மேடு அழுத்தத்தை குறைக்கிறது.",
        "society_en": "It encourages recycling awareness, responsible disposal, and resource-saving habits.",
        "society_ta": "இது மறுசுழற்சி விழிப்புணர்வு, பொறுப்பான அகற்றல், மற்றும் வளச் சேமிப்பு பழக்கங்களை ஊக்குவிக்கிறது.",
        "judge_note_en": "Paper segregation shows how a simple household action can support clean cities and circular resource use.",
        "judge_note_ta": "காகிதப் பிரிப்பு, ஒரு எளிய வீட்டு நடவடிக்கையே சுத்தமான நகரங்களுக்கும் சுற்றுசுழற்சி வளப் பயன்பாட்டிற்கும் எப்படி உதவுகிறது என்பதை காட்டுகிறது.",
        "verify_en": "Wet, laminated, or food-stained paper may need manual segregation.",
        "verify_ta": "நனைந்த, லமினேட் செய்யப்பட்ட அல்லது உணவு கறை பட்ட காகிதம் கைமுறையாக பிரிக்கப்பட வேண்டும்."
    },
    "plastic": {
        "name_en": "Plastic",
        "name_ta": "பிளாஸ்டிக்",
        "category_en": "Non-biodegradable",
        "category_ta": "உயிர்ச்சிதைவு அடையாத கழிவு",
        "stream_en": "Dry Recyclable Waste",
        "stream_ta": "உலர் மறுசுழற்சி கழிவு",
        "disposal_en": "Clean it if possible, keep it separate from wet waste, and send it for plastic recycling.",
        "disposal_ta": "முடிந்தால் சுத்தம் செய்து, ஈரக் கழிவிலிருந்து தனியாக வைத்திருந்து, பிளாஸ்டிக் மறுசுழற்சிக்கு அனுப்ப வேண்டும்.",
        "impact_en": "Plastic waste can persist in the environment for a long time, so segregation is important to reduce pollution.",
        "impact_ta": "பிளாஸ்டிக் நீண்ட காலம் சுற்றுச்சூழலில் நீடிக்கக்கூடும். எனவே மாசுபாட்டைக் குறைக்க இதை பிரித்துச் சேகரிப்பது மிகவும் அவசியம்.",
        "recycle_en": "It can be recycled into containers, benches, buckets, and other useful products depending on quality.",
        "recycle_ta": "தரத்தைப் பொறுத்து இதை டப்பா, பெஞ்ச், வாளி போன்ற பயனுள்ள பொருட்களாக மறுசுழற்சி செய்யலாம்.",
        "energy_en": "Plastic is not for composting. In some city systems, segregated low-value plastics may be sent for co-processing as fuel.",
        "energy_ta": "பிளாஸ்டிக் கம்போஸ்டுக்கு பொருந்தாது. சில நகர முறைமைகளில், குறைந்த மதிப்புள்ள பிரிக்கப்பட்ட பிளாஸ்டிக் எரிபொருள் இணை செயலாக்கத்திற்கு அனுப்பப்படலாம்.",
        "health_en": "Segregation reduces open dumping and burning risks, which helps protect public health.",
        "health_ta": "பிளாஸ்டிக்கை பிரித்துச் சேகரிப்பது திறந்த குப்பை கொட்டல் மற்றும் எரிப்பு அபாயத்தை குறைத்து, பொதுமக்கள் ஆரோக்கியத்தைப் பாதுகாக்க உதவுகிறது.",
        "smart_city_en": "Plastic segregation supports material recovery, cleaner streets, and better organized urban waste systems.",
        "smart_city_ta": "பிளாஸ்டிக் பிரிப்பு, பொருள் மீட்பு, சுத்தமான தெருக்கள் மற்றும் ஒழுங்கான நகர கழிவு அமைப்புகளை உருவாக்க உதவுகிறது.",
        "society_en": "This creates recycling value, improves cleanliness, and encourages responsible consumer behavior.",
        "society_ta": "இது மறுசுழற்சி மதிப்பை உருவாக்கி, சுத்தத்தைக் கூட்டி, பொறுப்பான நுகர்வோர் பழக்கத்தை ஊக்குவிக்கிறது.",
        "judge_note_en": "Plastic segregation is critical because it links environment, health, recycling economy, and smart-city cleanliness goals.",
        "judge_note_ta": "பிளாஸ்டிக் பிரிப்பு மிக முக்கியமானது. இது சுற்றுச்சூழல், ஆரோக்கியம், மறுசுழற்சி பொருளாதாரம் மற்றும் ஸ்மார்ட் நகர சுத்தம் ஆகியவற்றை ஒன்றிணைக்கிறது.",
        "verify_en": "If the object contains mixed materials like foil, food, or metal parts, verify manually.",
        "verify_ta": "இந்த பொருளில் ஃபோயில், உணவுப் பாகம் அல்லது உலோகப் பாகங்கள் கலந்து இருந்தால் கைமுறையாகச் சரிபார்க்கவும்."
    },
    "trash": {
        "name_en": "Trash",
        "name_ta": "பொது கழிவு",
        "category_en": "Mostly non-biodegradable or mixed waste",
        "category_ta": "பொதுவாக உயிர்ச்சிதைவு அடையாத அல்லது கலப்பு கழிவு",
        "stream_en": "Reject / Mixed Waste",
        "stream_ta": "மீதமுள்ள / கலப்பு கழிவு",
        "disposal_en": "This appears to be general mixed waste. It should be manually checked and separated as much as possible before final disposal.",
        "disposal_ta": "இது பொதுவான கலப்பு கழிவாகத் தெரிகிறது. இறுதி அகற்றுதலுக்கு முன் இயன்றவரை கைமுறையாகப் பிரித்துச் சரிபார்க்க வேண்டும்.",
        "impact_en": "Mixed waste reduces recycling efficiency and increases landfill burden.",
        "impact_ta": "கலப்பு கழிவு, மறுசுழற்சி திறனை குறைத்து, குப்பை மேடு சுமையை அதிகரிக்கிறது.",
        "recycle_en": "Low direct recycling value unless separated into reusable or recyclable components.",
        "recycle_ta": "மீண்டும் பயன்படுத்தக்கூடிய அல்லது மறுசுழற்சி செய்யக்கூடிய பகுதிகளாகப் பிரிக்காமல் இருந்தால் இதற்கு குறைந்த மறுசுழற்சி மதிப்பே இருக்கும்.",
        "energy_en": "This is not ideal for composting or renewable-energy recovery unless first segregated properly.",
        "energy_ta": "முதலில் சரியாகப் பிரிக்கப்படாவிட்டால் இது கம்போஸ்ட் அல்லது புதுப்பிக்கத்தக்க ஆற்றல் மீட்பிற்கு ஏற்றதல்ல.",
        "health_en": "Mixed waste can attract pests, create foul odour, and increase unhygienic conditions.",
        "health_ta": "கலப்பு கழிவு பூச்சிகளை ஈர்க்கும், துர்நாற்றத்தை உண்டாக்கும், மற்றும் அசுத்தமான சூழலை உருவாக்கும்.",
        "smart_city_en": "Reducing mixed waste is essential for efficient urban waste logistics and smart sanitation systems.",
        "smart_city_ta": "கலப்பு கழிவை குறைப்பது நகர கழிவு போக்குவரத்து திறன் மற்றும் ஸ்மார்ட் சுகாதார அமைப்புகளுக்கு அத்தியாவசியம்.",
        "society_en": "Better segregation at household level improves cleanliness, recovery value, and public health.",
        "society_ta": "வீட்டு மட்டத்திலேயே நல்ல பிரிப்பு நடந்தால் சுத்தம், பொருள் மீட்பு மதிப்பு மற்றும் பொது ஆரோக்கியம் மேம்படும்.",
        "judge_note_en": "Mixed waste is the real problem in cities. Our project helps reduce it at the source, which benefits health, renewable systems, and smart-city waste flow.",
        "judge_note_ta": "நகரங்களில் உண்மையான பிரச்சினை கலப்பு கழிவுதான். எங்கள் திட்டம் இதை மூலத்திலேயே குறைக்க உதவுவதால் ஆரோக்கியம், புதுப்பிக்கத்தக்க ஆற்றல் மற்றும் ஸ்மார்ட் நகர கழிவு ஒழுக்கு ஆகியவற்றிற்கு நன்மை தருகிறது.",
        "verify_en": "Manual segregation is strongly recommended for mixed waste.",
        "verify_ta": "கலப்பு கழிவிற்கு கைமுறை பிரிப்பு மிகவும் அவசியம்."
    }
}

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def predict_waste(image):
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(prediction))
    predicted_class = str(class_names[predicted_index]).lower().strip()
    confidence = float(np.max(prediction)) * 100

    return predicted_class, confidence

def get_lang_value(data, key_base, language):
    if language == "Tamil":
        return data.get(f"{key_base}_ta", "")
    return data.get(f"{key_base}_en", "")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
language = st.sidebar.selectbox(
    "Language / மொழி",
    ["English", "Tamil"]
)

t = ui_text[language]

st.title(t["title"])
st.markdown(t["subtitle"])

st.sidebar.markdown(f"### {t['language']}")
st.sidebar.success(t["footer"])

# --------------------------------------------------
# INPUT
# --------------------------------------------------
st.subheader(t["source"])
input_option = st.radio(
    t["source_option"],
    [t["upload"], t["camera"]]
)

image = None

if input_option == t["upload"]:
    uploaded_file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif input_option == t["camera"]:
    captured_image = st.camera_input(t["camera_label"])
    if captured_image is not None:
        image = Image.open(captured_image).convert("RGB")

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
if image is not None:
    st.image(image, caption="Selected Image", use_container_width=True)

    predicted_class, confidence = predict_waste(image)
    data = waste_data.get(predicted_class, waste_data["trash"])

    st.subheader(t["prediction"])
    st.success(f"{t['predicted']}: {get_lang_value(data, 'name', language)}")
    st.info(f"{t['confidence']}: {confidence:.2f}%")

    if confidence < 60:
        st.warning(t["low_conf"])

    st.subheader(t["category"])
    st.write(get_lang_value(data, "category", language))

    st.subheader(t["gov_stream"])
    st.write(get_lang_value(data, "stream", language))

    st.subheader(t["disposal"])
    st.write(get_lang_value(data, "disposal", language))

    st.subheader(t["impact"])
    st.write(get_lang_value(data, "impact", language))

    st.subheader(t["recycle"])
    st.write(get_lang_value(data, "recycle", language))

    st.subheader(t["energy"])
    st.write(get_lang_value(data, "energy", language))

    st.subheader(t["health"])
    st.write(get_lang_value(data, "health", language))

    st.subheader(t["smart_city"])
    st.write(get_lang_value(data, "smart_city", language))

    st.subheader(t["society"])
    st.write(get_lang_value(data, "society", language))

    st.subheader(t["judge_note"])
    st.write(get_lang_value(data, "judge_note", language))

    st.subheader(t["verify"])
    st.write(get_lang_value(data, "verify", language))

else:
    st.info(t["info_wait"])