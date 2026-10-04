import "./globals.css";

export const metadata = {
  title: "Photo Retrieval Evidence",
  description: "What users say when they cannot find a photo: corpus health, themes and evidence.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
