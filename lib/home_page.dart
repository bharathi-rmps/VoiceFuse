import 'dart:io';

import 'package:audioplayers/audioplayers.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sound/public/flutter_sound_recorder.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:hexcolor/hexcolor.dart';
import 'package:language_picker/language_picker_dropdown.dart';
import 'package:language_picker/languages.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'auth/sign_in.dart';
import 'methods/auth_method.dart';
import 'methods/storage_method.dart';

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String? inputLanguage;
  String? outputLanguage;
  final recorder = FlutterSoundRecorder();
  final audioPlayer = AudioPlayer();
  bool isPlaying = false;
  Duration duration = Duration.zero;
  Duration position = Duration.zero;
  File? inputAudioFile;
  bool isRecorderReady = false;
  String? url;
  bool isRecording = false;
  bool recordDone = false;

  Future record() async {
    if (!isRecorderReady) return;
    await recorder.startRecorder(toFile: 'audio');
  }

  Future stop() async {
    if (!isRecorderReady) return;
    final path = await recorder.stopRecorder();
    setState(() {
      inputAudioFile = File(path!);
    });
  }

  Future initRecorder() async {
    final status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      throw 'Microphone permission not granted';
    }
    await recorder.openRecorder();
    isRecorderReady = true;
    recorder.setSubscriptionDuration(
      const Duration(milliseconds: 500),
    );
  }

  @override
  void initState() {
    super.initState();
    initRecorder();
    audioPlayer.onPlayerStateChanged.listen((event) {
      setState(() {
        isPlaying = event == PlayerState.playing;
      });
    });
    audioPlayer.onDurationChanged.listen((newDuration) {
      setState(() {
        duration = newDuration;
      });
    });
    audioPlayer.onPositionChanged.listen((newPosition) {
      setState(() {
        position = newPosition;
      });
    });
  }

  @override
  void dispose() {
    recorder.closeRecorder();
    audioPlayer.dispose();
    super.dispose();
  }

  String formatTime(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    final hours = twoDigits(duration.inHours);
    final minutes = twoDigits(duration.inMinutes.remainder(60));
    final seconds = twoDigits(duration.inSeconds.remainder(60));
    return [if (duration.inHours > 0) hours, minutes, seconds].join(':');
  }

  Future setAudio() async {
    audioPlayer.setReleaseMode(ReleaseMode.loop);
    final snapshot = await FirebaseFirestore.instance
        .collection('users')
        .doc(FirebaseAuth.instance.currentUser!.uid)
        .get();

    url = snapshot.data()!['audioUrl'].toString();
    audioPlayer.setSourceUrl(url.toString());
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          automaticallyImplyLeading: false,
          actions: [
            ElevatedButton(
              onPressed: () async {
                AuthMethods().signOut();
                final prefs = await SharedPreferences.getInstance();
                prefs.setBool('isLoggedIn', false);
                Navigator.pushAndRemoveUntil(
                    context,
                    MaterialPageRoute(builder: (context) => const LoginPage()),
                    (route) => false);
              },
              child: const Text('Logout'),
            ),
          ],
        ),
        body: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 15.0),
          child: Column(
            children: [
              const SizedBox(
                height: 15,
              ),
              Container(
                decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(35.0),
                    color: HexColor('B2A4FF')),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 10),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      Expanded(
                        child: SizedBox(
                          height: 80,
                          width: 110,
                          child: LanguagePickerDropdown(
                              onValuePicked: (Language language) {
                            if (kDebugMode) {
                              print(language.name);
                            }
                            setState(() {
                              inputLanguage = language.name.toString();
                            });
                          }),
                        ),
                      ),
                      const SizedBox(
                        width: 30,
                      ),
                      Image.asset(
                        'assets/loop.png',
                        color: Colors.black,
                        height: 30,
                      ),
                      const SizedBox(
                        width: 30,
                      ),
                      Expanded(
                        child: SizedBox(
                          height: 80,
                          width: 110,
                          child: LanguagePickerDropdown(
                              onValuePicked: (Language language) {
                            if (kDebugMode) {
                              print(language.name);
                            }
                            setState(() {
                              outputLanguage = language.name.toString();
                            });
                          }),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(
                height: 80,
              ),
              StreamBuilder<RecordingDisposition>(
                stream: recorder.onProgress,
                builder: (context, snapshot) {
                  final duration = snapshot.hasData
                      ? snapshot.data!.duration
                      : Duration.zero;
                  String twoDigits(int n) => n.toString().padLeft(2, '0');
                  final twoDigitMinutes =
                      twoDigits(duration.inMinutes.remainder(60));
                  final twoDigitSeconds =
                      twoDigits(duration.inSeconds.remainder(60));
                  return Text(
                    '$twoDigitMinutes:$twoDigitSeconds',
                    style: const TextStyle(
                      fontSize: 80,
                      fontWeight: FontWeight.bold,
                    ),
                  );
                },
              ), // StreamBuilder
              const SizedBox(
                height: 10,
              ),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  GestureDetector(
                    onLongPress: () async {
                      HapticFeedback.heavyImpact();
                      setState(() {
                        isRecording = true;
                      });
                      await record();
                    },
                    onLongPressUp: () async {
                      HapticFeedback.heavyImpact();
                      setState(() {
                        isRecording = false;
                      });
                      if (recorder.isRecording) {
                        await stop();
                      }
                    },
                    child: Container(
                      height: 80,
                      width: 80,
                      decoration: BoxDecoration(
                          color: HexColor('E3DFFD'),
                          borderRadius: BorderRadius.circular(50)),
                      child: FittedBox(
                        child: Center(
                          child: Icon(
                            color: Colors.black,
                            isRecording ? Icons.stop : Icons.mic,
                            size: 50,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(
                height: 20,
              ),
              !isRecording
                  ? const Text('Hold to record')
                  : const Text('Release to stop recording'),
              const SizedBox(
                height: 10,
              ),
              ElevatedButton(
                  onPressed: () async {
                    if (inputAudioFile == null) {
                      Fluttertoast.showToast(
                          msg: 'Hold to record audio and translate');
                    }
                    if (inputAudioFile != null) {
                      setState(() {
                        recordDone = true;
                      });
                    }
                    String audioUrl = await StorageMethods()
                        .uploadAudioToStroage('inputAudio', inputAudioFile!);
                    await FirebaseFirestore.instance
                        .collection('users')
                        .doc(FirebaseAuth.instance.currentUser!.uid)
                        .update({
                      'audioUrl': audioUrl,
                      'inputLanguage': inputLanguage,
                      'outputLanguage': outputLanguage
                    });
                    setAudio();
                  },
                  child: const Text('Translate')),
              const SizedBox(
                height: 30,
              ),
              url != null
                  ? Column(
                      children: [
                        Slider(
                          min: 0,
                          max: duration.inSeconds.toDouble(),
                          value: position.inSeconds.toDouble(),
                          onChanged: (value) async {
                            final position = Duration(seconds: value.toInt());
                            await audioPlayer.seek(position);
                            await audioPlayer.resume();
                          },
                        ),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(formatTime(position)),
                            Text(formatTime(duration - position))
                          ],
                        ),
                        const SizedBox(
                          height: 20,
                        ),
                        const SizedBox(
                          height: 20,
                        ),
                        Container(
                          height: 120,
                          decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(30),
                              border: Border.all(color: HexColor('E3DFFD'))),
                          child: Row(
                            children: [
                              const Expanded(
                                  flex: 3,
                                  child: FittedBox(
                                    child: Text(
                                      'Translated text comes here',
                                      style: TextStyle(fontSize: 50),
                                    ),
                                  )),
                              VerticalDivider(
                                  color: HexColor('E3DFFD'), thickness: 1),
                              Expanded(
                                  flex: 1,
                                  child: CircleAvatar(
                                    radius: 35,
                                    child: IconButton(
                                      icon: Icon(isPlaying
                                          ? Icons.pause
                                          : Icons.play_arrow),
                                      iconSize: 50,
                                      onPressed: () async {
                                        if (isPlaying) {
                                          await audioPlayer.pause();
                                        } else {
                                          await audioPlayer.resume();
                                        }
                                      },
                                    ),
                                  ))
                            ],
                          ),
                        )
                      ],
                    )
                  : recordDone
                      ? const LinearProgressIndicator()
                      : const SizedBox()
            ],
          ),
        ));
  }
}
