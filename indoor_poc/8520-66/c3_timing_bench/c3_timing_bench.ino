// USB-powered bench only. Wire SDA=GPIO4, SCL=GPIO5, 3.3 V logic.
// Scan five address groups at startup. Time a 14-byte IMU read at 250 Hz.
#include <Arduino.h>
#include <Wire.h>
uint8_t imuAddress;
uint32_t nextTick, lastReport, lateByPeriod, readErrors, worstRead, peakLate, samples;

bool present(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Wire.begin(4, 5, 400000);
  Wire.setTimeOut(3);
  const uint8_t addresses[] = {0x68,0x69,0x29,0x44,0x45,0x76,0x77,0x23,0x5c};
  for (uint8_t address : addresses)
    if (present(address)) Serial.printf("I2C ACK 0x%02x\n", address);
  imuAddress = present(0x68) ? 0x68 : 0x69;
  Wire.beginTransmission(imuAddress);
  Wire.write(0x6b);
  Wire.write(0);
  if (Wire.endTransmission()) Serial.println("IMU wake failed");
  nextTick = micros() + 4000;
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now-nextTick) < 0) return;
  uint32_t late = now-nextTick;
  peakLate = max(peakLate, late);
  lateByPeriod += late/4000;
  nextTick += (1+late/4000)*4000;
  Wire.beginTransmission(imuAddress);
  Wire.write(0x3b);
  bool ok = Wire.endTransmission(false) == 0;
  if (ok) ok = Wire.requestFrom(imuAddress,(uint8_t)14) == 14;
  while (Wire.available()) Wire.read();
  if (!ok) ++readErrors;
  ++samples;
  worstRead = max(worstRead, uint32_t(micros()-now));
  if (uint32_t(millis()-lastReport) >= 1000) {
    Serial.printf("samples=%lu errors=%lu skipped=%lu max_late_us=%lu max_read_us=%lu\n",
                  (unsigned long)samples,(unsigned long)readErrors,(unsigned long)lateByPeriod,
                  (unsigned long)peakLate,(unsigned long)worstRead);
    samples=readErrors=lateByPeriod=peakLate=worstRead=0;
    lastReport=millis();
  }
}
