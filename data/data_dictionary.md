# Data Dictionary — PhiUSIIL Phishing URL Dataset

## Informasi Umum
| Parameter | Nilai |
|-----------|-------|
| **Nama Dataset** | PhiUSIIL Phishing URL Dataset |
| **Sumber** | UCI Machine Learning Repository |
| **Jumlah Baris** | 235.795 |
| **Jumlah Kolom** | 56 |
| **Target** | `label` (0 = Legitimate, 1 = Phishing) |
| **Format** | CSV |

## Deskripsi Fitur

### Kolom Non-Numerik (Akan di-drop saat preprocessing)
| No | Nama Kolom | Tipe | Deskripsi |
|----|-----------|------|-----------|
| 1 | FILENAME | string | Nama file sumber data |
| 2 | URL | string | URL lengkap |
| 3 | Domain | string | Nama domain dari URL |
| 4 | TLD | string | Top-Level Domain (.com, .org, dll.) |
| 5 | Title | string | Judul halaman web |

### Fitur URL-Based (Numerik)
| No | Nama Kolom | Tipe | Deskripsi |
|----|-----------|------|-----------|
| 6 | URLLength | int | Panjang karakter URL |
| 7 | DomainLength | int | Panjang karakter domain |
| 8 | IsDomainIP | binary | Apakah domain berupa IP address (0/1) |
| 9 | URLSimilarityIndex | float | Indeks kemiripan URL dengan domain |
| 10 | CharContinuationRate | float | Tingkat kelanjutan karakter berurutan |
| 11 | TLDLegitimateProb | float | Probabilitas TLD sebagai legitimate |
| 12 | URLCharProb | float | Probabilitas karakter URL |
| 13 | TLDLength | int | Panjang TLD |
| 14 | NoOfSubDomain | int | Jumlah subdomain |
| 15 | HasObfuscation | binary | Apakah URL mengandung obfuscation (0/1) |
| 16 | NoOfObfuscatedChar | int | Jumlah karakter terobfuskasi |
| 17 | ObfuscationRatio | float | Rasio obfuscation |
| 18 | NoOfLettersInURL | int | Jumlah huruf dalam URL |
| 19 | LetterRatioInURL | float | Rasio huruf dalam URL |
| 20 | NoOfDegitsInURL | int | Jumlah digit dalam URL |
| 21 | DegitRatioInURL | float | Rasio digit dalam URL |
| 22 | NoOfEqualsInURL | int | Jumlah tanda '=' dalam URL |
| 23 | NoOfQMarkInURL | int | Jumlah tanda '?' dalam URL |
| 24 | NoOfAmpersandInURL | int | Jumlah tanda '&' dalam URL |
| 25 | NoOfOtherSpecialCharsInURL | int | Jumlah karakter spesial lain |
| 26 | SpacialCharRatioInURL | float | Rasio karakter spesial |
| 27 | IsHTTPS | binary | Apakah menggunakan HTTPS (0/1) |

### Fitur Content-Based (Numerik)
| No | Nama Kolom | Tipe | Deskripsi |
|----|-----------|------|-----------|
| 28 | LineOfCode | int | Jumlah baris kode HTML |
| 29 | LargestLineLength | int | Panjang baris kode terpanjang |
| 30 | HasTitle | binary | Apakah halaman memiliki title tag (0/1) |
| 31 | DomainTitleMatchScore | float | Skor kecocokan domain-title |
| 32 | URLTitleMatchScore | float | Skor kecocokan URL-title |
| 33 | HasFavicon | binary | Apakah memiliki favicon (0/1) |
| 34 | Robots | binary | Apakah ada robots.txt (0/1) |
| 35 | IsResponsive | binary | Apakah halaman responsif (0/1) |
| 36 | NoOfURLRedirect | int | Jumlah URL redirect |
| 37 | NoOfSelfRedirect | int | Jumlah self-redirect |
| 38 | HasDescription | binary | Apakah ada meta description (0/1) |
| 39 | NoOfPopup | int | Jumlah popup |
| 40 | NoOfiFrame | int | Jumlah iFrame |
| 41 | HasExternalFormSubmit | binary | Apakah ada form submit eksternal (0/1) |
| 42 | HasSocialNet | binary | Apakah ada link media sosial (0/1) |
| 43 | HasSubmitButton | binary | Apakah ada tombol submit (0/1) |
| 44 | HasHiddenFields | binary | Apakah ada hidden field (0/1) |
| 45 | HasPasswordField | binary | Apakah ada password field (0/1) |
| 46 | Bank | binary | Apakah terkait perbankan (0/1) |
| 47 | Pay | binary | Apakah terkait pembayaran (0/1) |
| 48 | Crypto | binary | Apakah terkait cryptocurrency (0/1) |
| 49 | HasCopyrightInfo | binary | Apakah ada info copyright (0/1) |
| 50 | NoOfImage | int | Jumlah gambar |
| 51 | NoOfCSS | int | Jumlah file CSS |
| 52 | NoOfJS | int | Jumlah file JavaScript |
| 53 | NoOfSelfRef | int | Jumlah self-reference link |
| 54 | NoOfEmptyRef | int | Jumlah empty reference |
| 55 | NoOfExternalRef | int | Jumlah referensi eksternal |

### Target
| No | Nama Kolom | Tipe | Deskripsi |
|----|-----------|------|-----------|
| 56 | label | binary | **0** = Legitimate, **1** = Phishing |

## Batasan Penggunaan Data
- Dataset ini bersifat publik dan digunakan untuk tujuan akademik.
- Tidak mengandung data pribadi pengguna.
- Hasil prediksi tidak boleh digunakan sebagai satu-satunya dasar keputusan keamanan.
