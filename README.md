# Amazon Review Sentiment Analysis

Bu proje, Amazon yorumlarının duygu analizini yapmaktadır. Proje, **doğal dil işleme (NLP)** yöntemlerini kullanarak metin verilerinin analiz edilmesi, temizlenmesi ve sınıflandırılması işlemlerini içerir. Bu çalışmalar, **MIUUL**'de aldığım NLP eğitiminde edindiğim bilgiler ve tekniklerle gerçekleştirilmiştir.

## Proje Hakkında

Bu projede, Amazon ürün yorumları üzerinde çeşitli doğal dil işleme (NLP) teknikleri uygulanmıştır:

- **Text Preprocessing**: Veriler üzerinde büyük harf-küçük harf dönüşümü, noktalama işaretleri ve sayılar gibi gereksiz elemanlar temizlenmiştir.
- **Tokenization & Lemmatization**: Metinler, kelime parçalarına ayrılmış ve kelimeler köklerine indirgenmiştir.
- **Stopwords Removal**: Anlam taşıyan kelimeler dışında kalan kelimeler kaldırılmıştır.
- **Word Frequency Analysis & Visualization**: Metinlerdeki kelimelerin sıklıkları hesaplanmış ve görselleştirilmiştir.
- **Sentiment Analysis**: Yorumların duygu durumları analiz edilerek pozitif veya negatif etiketlenmiştir.
- **Feature Engineering**: Metin verisi sayısal verilere dönüştürülerek model için uygun hale getirilmiştir.

## Kullanılan Yöntemler ve Kütüphaneler

- **Pandas**: Verileri işlemek için
- **NumPy**: Sayısal işlemler için
- **NLTK**: Doğal dil işleme için
- **TextBlob**: Metin işleme ve duygu analizi için
- **WordCloud**: Metin görselleştirmesi için
- **Scikit-learn**: Makine öğrenmesi modellemeleri için
- **Logistic Regression & Random Forest**: Duygu analizinin sınıflandırılması için
- **Matplotlib**: Grafikler ve görselleştirmeler için

## Adımlar

1. **Veri Okuma ve İnceleme**:  
   Veriler, `amazon_reviews.csv` dosyasından okunup incelenmiştir.
   
2. **Veri Temizleme**:
   - **Case Folding**: Küçük harf kullanımıyla tutarlılık sağlanmıştır.
   - **Punctuation Removal**: Noktalama işaretleri temizlenmiştir.
   - **Sayılardan Arındırma**: Sayılar veriden çıkarılmıştır.
   - **Stopwords Kaldırma**: Gereksiz kelimeler (bağlaç, edat gibi) metinlerden temizlenmiştir.
   - **Rare Words Removal**: Nadiren kullanılan kelimeler veriden çıkarılmıştır.

3. **Sentiment Analysis**:
   - Metinlerin pozitif mi negatif mi olduğu `SentimentIntensityAnalyzer` aracılığıyla analiz edilmiştir.
   - Duygu skorlarına göre metinler etiketlenmiştir.

4. **Feature Engineering**:
   - `CountVectorizer` ve `TfidfVectorizer` kullanılarak metin verisi sayısal verilere dönüştürülmüştür.
   - N-gram analizleri yapılmıştır.

5. **Modelleme ve Değerlendirme**:
   - **Logistic Regression** ve **Random Forest** sınıflandırma algoritmaları ile modeller oluşturulmuş ve başarıları çapraz doğrulama ile değerlendirilmiştir.

