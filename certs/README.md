# Сертификат GigaChat API

`russian_trusted_root_ca.pem` — публичный корневой сертификат Russian Trusted Root CA. Он загружен 11.10.2026 по HTTPS с [официального адреса](https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt), указанного в [документации GigaChat](https://developers.sber.ru/docs/ru/gigachat/certificates).

SHA-256 сертификата: `D2:6D:2D:02:31:B7:C3:9F:92:CC:73:85:12:BA:54:10:35:19:E4:40:5D:68:B5:BD:70:3E:97:88:CA:8E:CF:31`.

Срок действия: 01.03.2022–27.02.2032 (UTC). Перед заменой проверяйте источник, срок и назначение CA.

Файл применяется через `LLM_CA_BUNDLE` только к HTTPS-запросам парсера. Проверка имени сервера и цепочки сертификатов остаётся включённой. Системное хранилище сертификатов не изменяется.
